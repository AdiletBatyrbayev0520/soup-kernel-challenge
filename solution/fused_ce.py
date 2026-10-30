"""Fused cross-entropy + TRL/Soup metrics over the 151,936-token vocabulary, without fp32 logits.

Base Soup keeps up to four fp32 [B, T, V] tensors alive (the .float() upcast in ForCausalLMLoss,
accelerate's fp32 output copy, log_softmax saved for backward, TRL's .contiguous() for accuracy),
about 1.2 GB each. Here only the fp16/bf16 logits from lm_head exist: one Triton pass per row gives
logsumexp, the loss, the entropy and argmax == label, and the backward overwrites the logits with
their gradient in place.

Idea from Unsloth's Fast_CrossEntropyLoss (unsloth/kernels/cross_entropy_loss.py: per-row lse
saved, softmax - onehot in backward); the online entropy/argmax pass and the Soup/TRL wiring are
ours. The metrics reach TRL and Soup's rewind log through Soup's own no-logits protocol
(soup_cli.utils.stream_loss_chunk.ChunkedStep, #1693), so their values stay the same.
"""

import torch
import triton
import triton.language as tl

from .machines import profile

IGNORE = -100


@triton.jit
def ce_stats_kernel(logits_ptr, stride, target_ptr, lse_ptr, loss_ptr, entropy_ptr, correct_ptr,
                    V, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = logits_ptr + row * stride
    target = tl.load(target_ptr + row)
    m = tl.full((), float("-inf"), tl.float32)  # running max
    s = tl.zeros((), tl.float32)  # sum exp(x - m)
    t = tl.zeros((), tl.float32)  # sum x * exp(x - m), for the entropy
    best = tl.full((), float("-inf"), tl.float32)
    best_idx = tl.zeros((), tl.int32)
    for start in range(0, V, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        inside = cols < V
        x = tl.load(base + cols, mask=inside, other=float("-inf")).to(tl.float32)
        block_max = tl.max(x, 0)
        block_idx = tl.argmax(x, 0)  # first index of the max, as torch.argmax
        take = block_max > best  # strict: an earlier block wins ties
        best_idx = tl.where(take, start + block_idx, best_idx)
        best = tl.maximum(best, block_max)
        m_new = tl.maximum(m, block_max)
        rescale = tl.exp(m - m_new)
        e = tl.exp(x - m_new)
        s = s * rescale + tl.sum(e, 0)
        t = t * rescale + tl.sum(tl.where(inside, x * e, 0.0), 0)
        m = m_new
    lse = m + tl.log(s)
    valid = target != -100
    x_target = tl.load(base + tl.where(valid, target, 0)).to(tl.float32)
    tl.store(lse_ptr + row, lse)
    tl.store(loss_ptr + row, tl.where(valid, lse - x_target, 0.0))
    tl.store(entropy_ptr + row, lse - t / s)
    tl.store(correct_ptr + row, (valid & (best_idx == target)).to(tl.int32))


@triton.jit
def ce_grad_kernel(logits_ptr, stride, target_ptr, lse_ptr, coef_ptr, V, BLOCK: tl.constexpr):
    """In place: logits <- (softmax - onehot(target)) * coef, 0 on rows without a label."""
    row = tl.program_id(0).to(tl.int64)
    base = logits_ptr + row * stride
    target = tl.load(target_ptr + row)
    lse = tl.load(lse_ptr + row)
    coef = tl.where(target != -100, tl.load(coef_ptr), 0.0)
    for start in range(0, V, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        inside = cols < V
        x = tl.load(base + cols, mask=inside, other=0.0).to(tl.float32)
        p = tl.exp(x - lse)
        p = tl.where(cols == target, p - 1.0, p)
        tl.store(base + cols, (p * coef).to(logits_ptr.dtype.element_ty), mask=inside)


def ce_stats(logits2d, target):
    """Per row of [N, V] logits: lse, loss (0 without a label), entropy, argmax == target."""
    n, v = logits2d.shape
    out = [torch.empty(n, device=logits2d.device, dtype=torch.float32) for _ in range(3)]
    correct = torch.empty(n, device=logits2d.device, dtype=torch.int32)
    p = profile()
    ce_stats_kernel[(n,)](logits2d, logits2d.stride(0), target, *out, correct, v,
                          BLOCK=p["ce_block"], num_warps=p["ce_warps"])
    return (*out, correct)


class FusedCrossEntropy(torch.autograd.Function):
    """sum of token CE / denom, as ForCausalLMLoss; also returns the per-token stats (no grad)."""

    @staticmethod
    def forward(ctx, logits, target, denom):
        logits2d = logits.view(-1, logits.shape[-1])
        lse, loss, entropy, correct = ce_stats(logits2d, target)
        ctx.save_for_backward(logits2d, target, lse, denom)
        ctx.shape = logits.shape
        ctx.mark_non_differentiable(loss, entropy, correct)
        return loss.sum() / denom, loss, entropy, correct

    @staticmethod
    def backward(ctx, grad, *_):
        logits2d, target, lse, denom = ctx.saved_tensors
        coef = (grad.float() / denom).reshape(1)
        p = profile()
        ce_grad_kernel[(logits2d.shape[0],)](logits2d, logits2d.stride(0), target, lse, coef,
                                             logits2d.shape[1], BLOCK=p["ce_block"],
                                             num_warps=p["ce_warps"])
        return logits2d.view(ctx.shape), None, None


def fused_causal_lm_loss(logits, labels, num_items_in_batch=None):
    """Loss as ForCausalLMLoss, plus Soup's ChunkedStep for TRL's metrics and the rewind log."""
    from soup_cli.utils.stream_loss_chunk import ChunkedStep, shift_for_next_token

    batch, seq = labels.shape
    target = shift_for_next_token(labels).reshape(-1).contiguous()
    valid = target != IGNORE
    if num_items_in_batch is None:
        denom = valid.sum().float()
    else:
        denom = torch.as_tensor(num_items_in_batch, device=logits.device).float()
    loss, token_loss, entropy, correct = FusedCrossEntropy.apply(logits.contiguous(), target, denom)
    step = ChunkedStep(token_entropy=entropy.view(batch, seq), token_loss=token_loss.view(batch, seq),
                       correct=correct.sum(), labelled=valid.sum())
    return loss, step


class _State:
    want = False  # set by compute_loss right around its forward
    step = None  # what that forward left for the metrics


STATE = _State()


def patch_fused_ce():
    """Qwen2ForCausalLM.forward returns loss without logits when our compute_loss asks for it;
    trl.SFTTrainer.compute_loss then logs TRL's metrics from the fused stats."""
    import trl
    from soup_cli.utils.stream_loss_chunk import STEP_ATTR, _log_trl_metrics
    from transformers import Trainer
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.qwen2 import modeling_qwen2

    model_forward = modeling_qwen2.Qwen2ForCausalLM.forward
    trainer_compute_loss = trl.SFTTrainer.compute_loss

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, labels=None, use_cache=None, logits_to_keep=0, **kwargs):
        if not (STATE.want and labels is not None and self.training
                and isinstance(logits_to_keep, int) and logits_to_keep == 0):
            return model_forward(self, input_ids, attention_mask, position_ids, past_key_values,
                                 inputs_embeds, labels, use_cache, logits_to_keep, **kwargs)
        STATE.want = False
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             position_ids=position_ids, past_key_values=past_key_values,
                             inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        logits = self.lm_head(outputs.last_hidden_state)
        loss, STATE.step = fused_causal_lm_loss(logits, labels, kwargs.get("num_items_in_batch"))
        return CausalLMOutputWithPast(loss=loss, logits=None, past_key_values=outputs.past_key_values,
                                      hidden_states=outputs.hidden_states,
                                      attentions=outputs.attentions)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        fused = (isinstance(model, torch.nn.Module) and getattr(self.model, "training", False)
                 and "labels" in inputs and "shift_labels" not in inputs
                 and not self.args.use_liger_kernel and not getattr(self, "aux_loss_enabled", False)
                 and getattr(self, "num_virtual_tokens", 0) == 0)
        if not fused:
            return trainer_compute_loss(self, model, inputs, return_outputs, num_items_in_batch)
        inputs.pop("_prediction_loss_only", None)  # as TRL's compute_loss does before it delegates
        inputs["use_cache"] = False
        STATE.want, STATE.step = True, None
        try:
            loss, outputs = Trainer.compute_loss(self, model, inputs, return_outputs=True,
                                                 num_items_in_batch=num_items_in_batch)
        finally:
            STATE.want = False
        step, STATE.step = STATE.step, None
        if step is None:
            raise RuntimeError("fused CE: the model forward did not take the fused path")
        setattr(self, STEP_ATTR, step)  # Soup's rewind recorder reads the token losses here
        _log_trl_metrics(self, "train", inputs, step)
        return (loss, outputs) if return_outputs else loss

    modeling_qwen2.Qwen2ForCausalLM.forward = forward
    trl.SFTTrainer.compute_loss = compute_loss

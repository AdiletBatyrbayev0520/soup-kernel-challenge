"""Causal attention without the padding mask, so SDPA can take its flash kernel.

TRL pads every batch on the right. Under causal attention a real token then only ever attends
to earlier real tokens, so the 2-D padding mask changes nothing for them; it only forces SDPA
onto the slower mem-efficient kernel with a [B, 1, T, T] mask and makes transformers repeat
k/v to all 12 heads. Without it, transformers calls SDPA with is_causal=True and enable_gqa.
Pad positions then see other pads, but they carry no label, no loss and no metric (TRL's
entropy is weighted by attention_mask). Batches that are not right-padded keep their mask.
Idea from Unsloth, which runs attention without a padding mask (flash / xformers causal).
"""

import torch


def right_padded(mask):
    """True when every row is 1...1 0...0 (one sync per step)."""
    return bool((mask[:, 1:] <= mask[:, :-1]).all())


def patch_causal_sdpa():
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2Model.forward

    def forward(self, input_ids=None, attention_mask=None, *args, **kwargs):
        if (self.training and isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 2
                and self.config._attn_implementation == "sdpa" and right_padded(attention_mask)):
            attention_mask = None
        return original(self, input_ids, attention_mask, *args, **kwargs)

    modeling_qwen2.Qwen2Model.forward = forward

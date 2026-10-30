"""Per-process time and memory of one training step: base Soup vs the `unsloth` branch.

    ./run_local.sh python results/process_report.py --machine 5090

Part 1 (isolated): the real 4-bit Qwen2.5-1.5B with LoRA q/v, loaded as Soup loads it (NF4 double
quant, prepare_model_for_kbit_training, autocast). Each process is run forward + backward on
layer 0 at the task's shape (batch 4 x 512 tokens), the base implementation against ours:
median CUDA-event time, peak allocated memory above the start, and the bytes it keeps for
backward (saved_tensors_hooks, parameters excluded). Per-step = per call x calls per step.
Part 2 (whole step): harness/_train.py --profile for base and solution, every CUDA kernel's GPU
time grouped by dashboard/server.py's categories.
Writes results/<machine>/process_report.csv and process_profile.csv.
"""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(REPO / "dashboard"))

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
B, T, LAYERS = 4, 512, 28
DT = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def load_model():
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=DT)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant, dtype=DT,
                                                 device_map={"": 0}, attn_implementation="sdpa")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules=["q_proj", "v_proj"],
                                             task_type="CAUSAL_LM"))
    model.train()
    return model


def saved_bytes(forward, params):
    """Bytes of the distinct non-parameter storages autograd keeps from one forward."""
    storages = {}

    def pack(t):
        s = t.untyped_storage()
        if s.data_ptr() not in params:
            storages[s.data_ptr()] = s.nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = forward()
    del out
    return sum(storages.values())


def measure(forward, backward, params, n=20):
    """forward() -> output; backward(output). Returns ms, peak MiB, saved MiB."""
    def step():
        backward(forward())

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    times = []
    for _ in range(n):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        step()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    step()
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - before) / 2**20
    return statistics.median(times), peak, saved_bytes(forward, params) / 2**20


def isolated(model):
    from transformers.loss.loss_utils import ForCausalLMLoss
    from transformers.models.qwen2 import modeling_qwen2 as q2
    from trl.trainer.utils import entropy_from_logits
    from soup_cli.trainer.rewind_hf import row_losses
    import bitsandbytes.functional as bnbF

    from solution.fused_ce import fused_causal_lm_loss
    from solution.nf4 import dequant_nf4
    from solution.rmsnorm import RMSNorm
    from solution.rope import RoPE
    from solution.swiglu import swiglu

    torch.manual_seed(0)
    inner = model.base_model.model
    layer = inner.model.layers[0]
    attn, mlp, norm = layer.self_attn, layer.mlp, layer.input_layernorm
    params = {p.untyped_storage().data_ptr() for p in model.parameters()}
    for p in model.parameters():  # the 4-bit weights hold their packed data in .data
        params.add(p.data.untyped_storage().data_ptr())
    H, V = inner.config.hidden_size, inner.config.vocab_size
    HQ, HK, D = inner.config.num_attention_heads, inner.config.num_key_value_heads, H // inner.config.num_attention_heads
    x = torch.randn(B, T, H, device="cuda", requires_grad=True)  # residual stream is fp32
    pos = torch.arange(T, device="cuda")[None].expand(B, T)
    cos, sin = inner.model.rotary_emb(x, pos)
    labels = torch.randint(0, V, (B, T), device="cuda")
    labels[:, : T // 3] = -100
    n_items = int((labels[:, 1:] != -100).sum())
    rows = []

    def add(process, calls, impl, fwd, note=""):
        def bwd(out):  # a gradient of ones into every output
            outs = (out,) if isinstance(out, torch.Tensor) else tuple(out)
            torch.autograd.backward(outs, [torch.ones_like(o) for o in outs])

        with torch.autocast("cuda", dtype=DT):
            ms, peak, saved = measure(fwd, bwd, params)
        rows.append({"process": process, "calls_per_step": calls, "impl": impl,
                     "ms_per_call": round(ms, 4), "ms_per_step": round(ms * calls, 2),
                     "peak_mib_per_call": round(peak, 1), "saved_mib_per_call": round(saved, 1),
                     "saved_mib_per_step": round(saved * calls, 1), "note": note})
        print(rows[-1], flush=True)

    # RMSNorm: 2 per layer + the final norm
    add("rmsnorm", 2 * LAYERS + 1, "base (HF Qwen2RMSNorm)", lambda: q2.Qwen2RMSNorm.forward(norm, x))
    add("rmsnorm", 2 * LAYERS + 1, "ours (Triton RMSNorm)",
        lambda: RMSNorm.apply(x, norm.weight, norm.variance_epsilon))

    # RoPE on q [B, 12, T, 128] and k [B, 2, T, 128], fp32 views as Qwen2Attention makes them
    q = torch.randn(B, T, HQ, D, device="cuda").transpose(1, 2).requires_grad_()
    k = torch.randn(B, T, HK, D, device="cuda").transpose(1, 2).requires_grad_()
    hf_rope = q2.apply_rotary_pos_emb
    add("rope", LAYERS, "base (HF apply_rotary_pos_emb)", lambda: hf_rope(q, k, cos, sin))
    add("rope", LAYERS, "ours (Triton RoPE)", lambda: RoPE.apply(q, k, cos, sin))

    # MLP: gate/up/down 4-bit matmuls + SwiGLU (bnb dequant on both sides: see nf4 below)
    hf_mlp = q2.Qwen2MLP.forward
    add("mlp (gate/up/down + act)", LAYERS, "base (HF, fp32 activations)", lambda: hf_mlp(mlp, x))
    add("mlp (gate/up/down + act)", LAYERS, "example SwiGLU (fp32 activations)",
        lambda: mlp.down_proj(swiglu(mlp.gate_proj(x), mlp.up_proj(x))))
    add("mlp (gate/up/down + act)", LAYERS, "ours (compute dtype + SwiGLU)",
        lambda: mlp.down_proj(swiglu(mlp.gate_proj(x.to(DT)), mlp.up_proj(x.to(DT)))))

    # attention projections q/k/v (+LoRA on q, v) and o: unchanged by us except the dequant
    a_in = torch.randn(B, T, H, device="cuda", dtype=DT, requires_grad=True)
    add("attention projections (q/k/v/o + LoRA q,v)", LAYERS, "base = ours (bnb dequant)",
        lambda: (attn.q_proj(x), attn.k_proj(x), attn.v_proj(x), attn.o_proj(a_in)))

    # attention core: SDPA with the padding mask, as HF's sdpa path (repeat_kv to 12 heads)
    qa = torch.randn(B, HQ, T, D, device="cuda", dtype=DT, requires_grad=True)
    ka = torch.randn(B, HK, T, D, device="cuda", dtype=DT, requires_grad=True)
    va = torch.randn(B, HK, T, D, device="cuda", dtype=DT, requires_grad=True)
    mask = torch.ones(T, T, device="cuda", dtype=torch.bool).tril()[None, None].expand(B, 1, T, T)
    rep = HQ // HK
    add("attention core (SDPA)", LAYERS, "base = ours",
        lambda: torch.nn.functional.scaled_dot_product_attention(
            qa, ka.repeat_interleave(rep, 1), va.repeat_interleave(rep, 1), attn_mask=mask))

    # NF4 dequant of the 7 weights of a layer, once forward and once backward
    weights = [m.base_layer.weight if hasattr(m, "base_layer") else m.weight
               for m in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj,
                         mlp.gate_proj, mlp.up_proj, mlp.down_proj)]
    for impl, fn in (("base (bitsandbytes, 2 kernels)", lambda w: bnbF.dequantize_4bit(w.data, w.quant_state)),
                     ("ours (Triton, 1 kernel)", lambda w: dequant_nf4(
                         w.data, w.quant_state.absmax, w.quant_state.state2.code,
                         w.quant_state.state2.absmax, w.quant_state.offset,
                         w.quant_state.blocksize, w.quant_state.shape, w.quant_state.dtype))):
        for _ in range(3):
            [fn(w) for w in weights]
        torch.cuda.synchronize()
        times = []
        for _ in range(20):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            [fn(w) for w in weights]
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        ms = statistics.median(times)
        rows.append({"process": "nf4 dequant (7 weights)", "calls_per_step": 2 * LAYERS,
                     "impl": impl, "ms_per_call": round(ms, 4), "ms_per_step": round(ms * 2 * LAYERS, 2),
                     "peak_mib_per_call": None, "saved_mib_per_call": 0.0,
                     "saved_mib_per_step": 0.0, "note": "transient W buffers only"})
        print(rows[-1], flush=True)

    # lm_head + loss + TRL entropy/accuracy + Soup rewind row losses
    h = torch.randn(B, T, H, device="cuda", requires_grad=True)
    head = inner.lm_head

    def base_head():
        logits = head(h)
        loss = ForCausalLMLoss(logits, labels, V, num_items_in_batch=n_items)
        logits32 = logits.float()  # accelerate's convert_outputs_to_fp32
        with torch.no_grad():
            entropy_from_logits(logits32)
            (logits32[:, :-1].contiguous().argmax(-1) == labels[:, 1:]).sum()
            row_losses(logits32, labels)
        return loss

    add("lm_head + cross-entropy + metrics", 1, "base (HF loss + TRL + Soup rewind)", base_head)
    add("lm_head + cross-entropy + metrics", 1, "ours (fused CE + stats)",
        lambda: fused_causal_lm_loss(head(h), labels, n_items)[0])
    return rows


def whole_step(out):
    """GPU ms per kernel category per step, base and solution, from harness/_train.py --profile."""
    from common import OFFLINE_ENV, read_json, run_logged
    from run_task import make_config
    import server

    rows = []
    for side in ("base", "solution"):
        run_dir = out / f"profile_{side}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = make_config(run_dir, REPO / "data" / "train.jsonl", 1)
        cmd = [sys.executable, str(REPO / "harness" / "_train.py"), "--config", str(cfg),
               "--metrics", str(run_dir / "metrics.json"), "--profile", str(run_dir / "profile.json")]
        cmd += ["--solution"] if side == "solution" else []
        env = dict(OFFLINE_ENV, SOUP_DB_PATH=str(run_dir / "experiments.db"),
                   TRITON_CACHE_DIR=str(run_dir / "triton_cache"))
        code, *_ = run_logged(cmd, run_dir / "train.log", 900, env)
        prof = read_json(run_dir / "profile.json", None)
        if code or not prof:
            raise SystemExit(f"profile {side} failed, see {run_dir / 'train.log'}")
        files = read_json(run_dir / "metrics.json", {}).get("triton_files", {})
        own = {n for n, f in files.items() if f.startswith("solution/")}
        cats, backward = server.categorize(prof["kernels"], own)
        steps = prof["active_steps"]
        row = {"side": side, "gpu_ms_per_step": round(prof["total_gpu_us"] / 1000 / steps, 2),
               **{k: round(v / steps, 2) for k, v in cats.items()},
               "backward_named": round(backward / steps, 2)}
        rows.append(row)
        print(row, flush=True)
    return rows


def write(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"written: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", default="5090")
    args = parser.parse_args()
    out_dir = REPO / "results" / args.machine
    out_dir.mkdir(exist_ok=True)
    write(out_dir / "process_profile.csv", whole_step(REPO / "runs" / "process_report"))
    model = load_model()
    write(out_dir / "process_report.csv", isolated(model))


if __name__ == "__main__":
    main()

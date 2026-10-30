# Reference: real Unsloth vs base Soup vs our `unsloth` branch — RTX 5090, 2026-10-31

Branch `unsloth-reference`. Raw rows: `results/5090/unsloth_reference.csv` (batch
`20261031-012429`); logs: `runs/unsloth_reference/20261031-012429/`. Reproduce with:

    ./run_local.sh python results/unsloth_reference.py --unsloth-python "$PWD/../.venv-unsloth/bin/python"

All four runs were measured in one session with the harness's own code: `harness/_train.py`
(step callbacks), `common.run_logged` (wall time and nvidia-smi peak from outside the process,
offline, empty compiler caches) and `harness/eval_adapter.py` (held-out loss, plain
transformers + peft). Unsloth runs use `results/_train_unsloth.py`, which is `_train.py` with
its `sys.modules["unsloth"] = None` block swapped for `import unsloth`. An untimed online Unsloth
warm-up ran first: 31.5 s, for downloads and Unsloth's compile cache.

| run | LoRA targets | grad. checkpointing | wall time | train loop (64 steps) | median step | peak GPU (nvidia-smi) | torch alloc | eval loss | vs base: time / memory |
|---|---|---|---|---|---|---|---|---|---|
| base Soup | q, v | off | 14.63 s | 8.86 s | 115 ms | 15,765 MiB | 13,340 MiB | 2.05317 | 1.00 / 1.00 |
| **ours** (`unsloth` branch) | q, v | off | **13.43 s** | 7.60 s | 89 ms | 8,272 MiB | 6,502 MiB | 2.05311 | **1.089x / 1.906x** |
| Unsloth `backend: unsloth` | all 7 | off | 17.69 s | **6.82 s** | **75 ms** | 8,090 MiB | 6,381 MiB | 1.92490 | 0.827x / 1.949x |
| Unsloth + `gradient_checkpointing: true` | all 7 | Unsloth's offloaded | 20.39 s | 8.93 s | 107 ms | **3,819 MiB** | 2,099 MiB | 1.92499 | 0.718x / **4.128x** |

## What it shows

- **Whole command:** Unsloth is slower end to end, 17.7 s vs 14.6 s for base and 13.4 s for us.
  Its training loop is the fastest (6.8 s), but importing and patching Unsloth plus its loader
  add about 9 s more startup than Soup's transformers path. The harness scores the whole
  command, so here our branch beats Unsloth on time.
- **Memory without checkpointing:** Unsloth's peak is 8,090 MiB, about the same as ours
  (8,272 MiB). It gets there while training LoRA on all 7 projections.
- **Memory with Unsloth's offloaded checkpointing:** 3,819 MiB (4.1x less than base), at
  0.72x speed. That switch is forbidden in the challenge. Our own recompute/offload would be
  allowed and could recover part of the gap.
- **Quality isn't comparable:** Unsloth's eval loss is 6.2% lower because it trains LoRA on
  all 7 modules (more trainable parameters), which is a different task. Its first logged
  training loss is also 3.01 vs 2.057 for base. So Unsloth's loss logging or batching
  differs from Soup's transformers path (normalisation or which tokens are counted). I didn't
  dig into it, and check 6 would fail on it.

## Caveats

- Unsloth runs in its own venv `../.venv-unsloth` (outside the repo) with a newer stack:
  unsloth 2026.10.11, torch 2.14.1+cu130, transformers 5.17.0, peft 0.21.2, triton 3.8.0. The
  main venv has torch 2.15.0, transformers 5.16.1, peft 0.20.0 and triton 3.9.0.
- Unsloth's xformers wheel was built for torch 2.10 and failed to load, so Unsloth fell back to
  PyTorch SDPA attention. A matching xformers might make its loop slightly faster.
- One run per configuration on a 5090 in bf16. The jury's T4 runs fp16, and the ratios there
  will differ.
- This branch is a measurement only. The banned `unsloth` package is never imported by
  `solution/` or by the base/solution harness runs.

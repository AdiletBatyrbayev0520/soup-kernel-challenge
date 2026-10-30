"""Per-GPU launch parameters for our kernels.

The profile comes from SOUP_MACHINE (t4, 3090, 4090, 5090) when it is set, else from the GPU's
name, else from its compute capability. The jury's Kaggle run sets nothing and gets "t4".
"""

import os

import torch

# ce_block: vocab columns per program step in the cross-entropy kernels; *_warps: num_warps.
# 5090: measured (CE stats 1538 GB/s, grad 1406 GB/s on bf16 [2048, 151936]); the rest are
# extrapolated from it by memory bandwidth and SM size: re-sweep on those cards.
PROFILES = {
    "t4": dict(arch="sm_75 Turing", ce_block=2048, ce_warps=8, row_warps=4, dequant_block=1024),
    "3090": dict(arch="sm_86 Ampere", ce_block=4096, ce_warps=8, row_warps=4, dequant_block=1024),
    "4090": dict(arch="sm_89 Ada", ce_block=4096, ce_warps=8, row_warps=4, dequant_block=1024),
    "5090": dict(arch="sm_120 Blackwell", ce_block=4096, ce_warps=4, row_warps=2, dequant_block=2048),
}
BY_CAPABILITY = {(7, 5): "t4", (8, 6): "3090", (8, 9): "4090", (12, 0): "5090"}


def detect():
    name = os.environ.get("SOUP_MACHINE", "").lower()
    if name:
        if name not in PROFILES:
            raise ValueError(f"SOUP_MACHINE={name!r}, expected one of {', '.join(PROFILES)}")
        return name
    if not torch.cuda.is_available():
        return "t4"
    gpu = torch.cuda.get_device_name().upper()
    for key in PROFILES:
        if key.upper() in gpu:
            return key
    return BY_CAPABILITY.get(torch.cuda.get_device_capability(), "t4")


MACHINE = None


def profile():
    global MACHINE
    MACHINE = MACHINE or detect()
    return PROFILES[MACHINE]

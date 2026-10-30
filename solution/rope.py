"""Rotary embedding of q and k in one Triton launch, forward and backward.

HF's apply_rotary_pos_emb is about ten element-wise kernels (two rotate_half copies, four muls, two
adds) and keeps q, k and their rotated copies alive for backward; here one program rotates every
head of one token, and backward is the same kernel with -sin, saving only cos and sin.
Idea from Unsloth's Fast_RoPE_Embedding (unsloth/kernels/rope_embedding.py: q and k together,
backward = rotation by -theta); the strided kernel is ours.
"""

import torch
import triton
import triton.language as tl

from .machines import profile


@triton.jit
def _rotate_heads(x_ptr, o_ptr, x_head_stride, o_head_stride, c, s,
                  H: tl.constexpr, H_P: tl.constexpr, HALF: tl.constexpr):
    heads = tl.arange(0, H_P)[:, None]
    d = tl.arange(0, HALF)[None, :]
    inside = heads < H
    x1 = tl.load(x_ptr + heads * x_head_stride + d, mask=inside, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + heads * x_head_stride + d + HALF, mask=inside, other=0.0).to(tl.float32)
    # x * cos + rotate_half(x) * sin, rotate_half(x) = (-x2, x1); halves of cos/sin are equal
    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s
    dtype = o_ptr.dtype.element_ty
    tl.store(o_ptr + heads * o_head_stride + d, o1.to(dtype), mask=inside)
    tl.store(o_ptr + heads * o_head_stride + d + HALF, o2.to(dtype), mask=inside)


@triton.jit
def rope_kernel(q_ptr, k_ptr, qo_ptr, ko_ptr, cos_ptr, sin_ptr, S,
                q_b, q_h, q_s, k_b, k_h, k_s, qo_b, qo_h, qo_s, ko_b, ko_h, ko_s, c_b, c_s,
                sign, HQ: tl.constexpr, HK: tl.constexpr, HQ_P: tl.constexpr,
                HK_P: tl.constexpr, HALF: tl.constexpr):
    pid = tl.program_id(0)
    b = (pid // S).to(tl.int64)
    t = (pid % S).to(tl.int64)
    d = tl.arange(0, HALF)[None, :]
    c = tl.load(cos_ptr + b * c_b + t * c_s + d).to(tl.float32)
    s = tl.load(sin_ptr + b * c_b + t * c_s + d).to(tl.float32) * sign
    _rotate_heads(q_ptr + b * q_b + t * q_s, qo_ptr + b * qo_b + t * qo_s, q_h, qo_h, c, s,
                  HQ, HQ_P, HALF)
    _rotate_heads(k_ptr + b * k_b + t * k_s, ko_ptr + b * ko_b + t * ko_s, k_h, ko_h, c, s,
                  HK, HK_P, HALF)


def rope(q, k, cos, sin, sign):
    """q, k: [B, H, S, D] with any strides (last dim contiguous); cos, sin: [B or 1, S, D]."""
    B, HQ, S, D = q.shape
    HK = k.shape[1]
    qo, ko = torch.empty_like(q), torch.empty_like(k)  # same layout as the input, as HF's muls
    cos, sin = cos.expand(B, S, D), sin.expand(B, S, D)
    rope_kernel[(B * S,)](
        q, k, qo, ko, cos, sin, S,
        q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
        qo.stride(0), qo.stride(1), qo.stride(2), ko.stride(0), ko.stride(1), ko.stride(2),
        cos.stride(0), cos.stride(1), sign,
        HQ=HQ, HK=HK, HQ_P=triton.next_power_of_2(HQ), HK_P=triton.next_power_of_2(HK),
        HALF=D // 2, num_warps=profile()["row_warps"],
        enable_fp_fusion=False)  # HF rounds the products and the sum separately
    return qo, ko


class RoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        ctx.save_for_backward(cos, sin)
        return rope(q, k, cos, sin, 1.0)

    @staticmethod
    def backward(ctx, dq, dk):
        cos, sin = ctx.saved_tensors
        dq, dk = rope(dq, dk, cos, sin, -1.0)  # the transpose of a rotation is the rotation by -theta
        return dq, dk, None, None


def _ours(q, k, cos, sin, unsqueeze_dim):
    if unsqueeze_dim != 1 or not (q.is_cuda and q.dim() == 4 and k.dim() == 4 and cos.dim() == 3):
        return False
    D = q.shape[-1]
    return (D % 2 == 0 and triton.next_power_of_2(D // 2) == D // 2 and k.shape[-1] == D
            and q.stride(-1) == 1 and k.stride(-1) == 1 and cos.stride(-1) == 1
            and sin.stride() == cos.stride() and cos.shape[-2] == q.shape[2]
            and q.shape[0] == k.shape[0] and q.shape[2] == k.shape[2]
            and q.dtype == k.dtype and cos.dtype == sin.dtype
            and cos.shape[0] in (1, q.shape[0]))


def patch_rope():
    """Attention calls apply_rotary_pos_emb through the module global: replace that."""
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.apply_rotary_pos_emb

    def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
        if not _ours(q, k, cos, sin, unsqueeze_dim):
            return original(q, k, cos, sin, unsqueeze_dim)
        if cos.dtype != q.dtype:  # HF promotes; keep it simple and exact
            return original(q, k, cos, sin, unsqueeze_dim)
        return RoPE.apply(q, k, cos, sin)

    modeling_qwen2.apply_rotary_pos_emb = apply_rotary_pos_emb

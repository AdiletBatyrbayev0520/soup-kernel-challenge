"""Qwen2RMSNorm forward and backward in one Triton launch each.

HF's norm is five fp32 element-wise kernels forward (and more backward) and keeps several [N, H]
fp32 intermediates for backward; here the forward saves only x and one fp32 inverse RMS per row.
Same rounding as HF: normalize in fp32, round to x's dtype, multiply by the (fp32) weight.
Idea from Unsloth's Fast_RMS_Layernorm (unsloth/kernels/rms_layernorm.py: save inv_var, dX from
it in one pass); the kernels are ours.
"""

import torch
import triton
import triton.language as tl

from .machines import profile


@triton.jit
def rmsnorm_fwd_kernel(x_ptr, w_ptr, y_ptr, inv_ptr, H, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    inside = cols < H
    x = tl.load(x_ptr + row * H + cols, mask=inside, other=0.0).to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(x * x, 0) / H + eps)  # as torch.rsqrt(mean(x^2) + eps)
    normed = (x * inv).to(x_ptr.dtype.element_ty).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=inside, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * H + cols, (w * normed).to(y_ptr.dtype.element_ty), mask=inside)
    tl.store(inv_ptr + row, inv)


@triton.jit
def rmsnorm_bwd_kernel(dy_ptr, x_ptr, w_ptr, inv_ptr, dx_ptr, H, BLOCK: tl.constexpr):
    """dx = inv * (g - n * mean(g * n)), g = dy * w, n = x * inv."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    inside = cols < H
    dy = tl.load(dy_ptr + row * H + cols, mask=inside, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * H + cols, mask=inside, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=inside, other=0.0).to(tl.float32)
    inv = tl.load(inv_ptr + row)
    g = dy * w
    n = x * inv
    dx = inv * (g - n * (tl.sum(g * n, 0) / H))
    tl.store(dx_ptr + row * H + cols, dx.to(dx_ptr.dtype.element_ty), mask=inside)


class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        shape, H = x.shape, x.shape[-1]
        x2 = x.reshape(-1, H).contiguous()
        y = torch.empty(x2.shape, dtype=torch.result_type(x, weight), device=x.device)
        inv = torch.empty(x2.shape[0], dtype=torch.float32, device=x.device)
        ctx.block, ctx.warps = triton.next_power_of_2(H), profile()["row_warps"]
        rmsnorm_fwd_kernel[(x2.shape[0],)](x2, weight, y, inv, H, eps, BLOCK=ctx.block,
                                           num_warps=ctx.warps)
        ctx.save_for_backward(x2, weight, inv)
        ctx.shape = shape
        return y.view(shape)

    @staticmethod
    def backward(ctx, dy):
        x2, weight, inv = ctx.saved_tensors
        H = x2.shape[-1]
        dy2 = dy.reshape(-1, H).contiguous()
        dx = torch.empty_like(x2)
        rmsnorm_bwd_kernel[(x2.shape[0],)](dy2, x2, weight, inv, dx, H, BLOCK=ctx.block,
                                           num_warps=ctx.warps)
        dw = None
        if ctx.needs_input_grad[1]:  # frozen in this task; kept correct for other uses
            normed = (x2.float() * inv[:, None]).to(x2.dtype)
            dw = (dy2.float() * normed.float()).sum(0).to(weight.dtype)
        return dx.view(ctx.shape), dw, None


def patch_rmsnorm():
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2RMSNorm.forward

    def forward(self, hidden_states):
        weight = getattr(self, "weight", None)
        if not (isinstance(weight, torch.Tensor) and hidden_states.is_cuda
                and hidden_states.shape[-1] == weight.numel() <= 8192):
            return original(self, hidden_states)
        return RMSNorm.apply(hidden_states, weight, self.variance_epsilon)

    modeling_qwen2.Qwen2RMSNorm.forward = forward

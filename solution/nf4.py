"""NF4 double-quant dequantization in one Triton launch, for every frozen 4-bit linear.

bitsandbytes dequantizes a nested-quant weight in two kernels (8-bit absmax -> fp32 absmax, then
4-bit codes -> weights) in both the forward (M > 1536 takes the dequant + F.linear fallback) and
the backward (grad @ W). Here one kernel reads the packed codes, the 8-bit absmax, its fp32
scales and offset, and writes W directly. Idea from Unsloth's fast_dequantize
(unsloth/kernels/utils.py), which calls bnb's two kernels into a reused buffer; the fused kernel
is ours.
"""

import torch
import triton
import triton.language as tl

from .machines import profile

NESTED_BLOCK = 256  # bnb's nested (absmax) blocksize: the only one it supports


@triton.jit
def nf4_dequant_kernel(packed_ptr, absmax8_ptr, absmax_code_ptr, absmax_scale_ptr, offset,
                       nf4_ptr, out_ptr, n, BLOCKSIZE: tl.constexpr, NESTED: tl.constexpr,
                       BLOCK: tl.constexpr):
    idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    inside = idx < n
    byte = tl.load(packed_ptr + idx // 2, mask=inside, other=0).to(tl.int32)
    code = tl.where(idx % 2 == 0, byte >> 4, byte & 15)  # high nibble first, as bnb packs
    block = idx // BLOCKSIZE
    absmax = tl.load(absmax_code_ptr + tl.load(absmax8_ptr + block, mask=inside, other=0).to(tl.int32))
    absmax = absmax * tl.load(absmax_scale_ptr + block // NESTED, mask=inside, other=0.0) + offset
    w = tl.load(nf4_ptr + code) * absmax
    tl.store(out_ptr + idx, w.to(out_ptr.dtype.element_ty), mask=inside)


_NF4 = {}


def dequant_nf4(packed, absmax8, absmax_code, absmax_scale, offset, blocksize, shape, dtype):
    """W [shape] in dtype from bnb's packed NF4 codes and nested 8-bit absmax."""
    device = packed.device
    if device not in _NF4:
        from bitsandbytes.functional import get_4bit_type
        _NF4[device] = get_4bit_type("nf4", device=device).float().contiguous()
    out = torch.empty(shape, dtype=dtype, device=device)
    n = out.numel()
    block = profile()["dequant_block"]
    nf4_dequant_kernel[(triton.cdiv(n, block),)](
        packed.contiguous(), absmax8, absmax_code, absmax_scale, float(offset), _NF4[device], out, n,
        BLOCKSIZE=blocksize, NESTED=NESTED_BLOCK, BLOCK=block, num_warps=4)
    return out


def _ours(packed, quant_type, absmax8, nested_block):
    return packed.is_cuda and quant_type == "nf4" and absmax8 is not None and \
        absmax8.dtype == torch.uint8 and nested_block == NESTED_BLOCK


def patch_nf4_dequant():
    import bitsandbytes.functional as bnb_functional
    from bitsandbytes.backends.cuda import ops

    fallback = ops._dequant_linear_fallback
    dequantize_4bit = bnb_functional.dequantize_4bit

    def dequant_linear(A, B, shapeB, absmax, blocksize, quant_type, bias=None, absmax_8bit=None,
                       absmax_code=None, absmax_offset=None):
        if not _ours(B, quant_type, absmax_8bit, NESTED_BLOCK):
            return fallback(A, B, shapeB, absmax, blocksize, quant_type, bias, absmax_8bit,
                            absmax_code, absmax_offset)
        W = dequant_nf4(B, absmax_8bit, absmax_code, absmax, absmax_offset, blocksize, shapeB, A.dtype)
        return torch.nn.functional.linear(A, W, bias)

    def dequant(A, quant_state=None, absmax=None, out=None, blocksize=None, quant_type="fp4"):
        qs = quant_state
        if qs is None or out is not None or not qs.nested or A.shape[0] == 1 or \
                not _ours(A, qs.quant_type, qs.absmax, qs.state2.blocksize):
            return dequantize_4bit(A, quant_state, absmax, out, blocksize, quant_type)
        return dequant_nf4(A, qs.absmax, qs.state2.code, qs.state2.absmax, qs.offset, qs.blocksize,
                           qs.shape, qs.dtype)

    ops._dequant_linear_fallback = dequant_linear
    bnb_functional.dequantize_4bit = dequant

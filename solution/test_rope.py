"""Our RoPE against transformers' apply_rotary_pos_emb, forward and backward."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("cos_batch", [1, 4])
def test_rope_matches_hf(dtype, cos_batch):
    from transformers.models.qwen2 import modeling_qwen2

    from solution.rope import RoPE, _ours

    torch.manual_seed(0)
    B, S, HQ, HK, D = 4, 77, 12, 2, 128
    # as Qwen2Attention: proj(x).view(B, S, H, D).transpose(1, 2), a non-contiguous view
    q = torch.randn(B, S, HQ, D, device="cuda", dtype=dtype).transpose(1, 2).requires_grad_()
    k = torch.randn(B, S, HK, D, device="cuda", dtype=dtype).transpose(1, 2).requires_grad_()
    freqs = torch.rand(cos_batch, S, D // 2, device="cuda") * 50
    emb = torch.cat((freqs, freqs), -1)
    cos, sin = emb.cos().to(dtype), emb.sin().to(dtype)
    gq, gk = torch.randn_like(q), torch.randn_like(k)
    assert _ours(q, k, cos, sin, 1)

    rq, rk = modeling_qwen2.apply_rotary_pos_emb(q, k, cos, sin)
    torch.autograd.backward((rq, rk), (gq, gk))
    ref_dq, ref_dk = q.grad.clone(), k.grad.clone()
    q.grad = k.grad = None
    oq, ok = RoPE.apply(q, k, cos, sin)
    torch.autograd.backward((oq, ok), (gq, gk))

    tol = {torch.float32: 1e-6, torch.bfloat16: 1e-2, torch.float16: 1e-3}[dtype]
    for ours, ref in ((oq, rq), (ok, rk), (q.grad, ref_dq), (k.grad, ref_dk)):
        torch.testing.assert_close(ours, ref, rtol=tol, atol=tol)

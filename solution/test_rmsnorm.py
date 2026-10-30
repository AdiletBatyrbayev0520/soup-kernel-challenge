"""Our RMSNorm against transformers' Qwen2RMSNorm, forward and backward."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("x_dtype,w_dtype", [(torch.float32, torch.float32),
                                             (torch.bfloat16, torch.float32),
                                             (torch.float16, torch.float16)])
@pytest.mark.parametrize("shape", [(4, 512, 1536), (3, 7, 1000)])
def test_rmsnorm_matches_hf(x_dtype, w_dtype, shape):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    from solution.rmsnorm import RMSNorm

    torch.manual_seed(0)
    norm = Qwen2RMSNorm(shape[-1], eps=1e-6).cuda().to(w_dtype)
    norm.weight.data.uniform_(0.5, 1.5)
    norm.weight.requires_grad_(False)
    x = torch.randn(shape, device="cuda", dtype=x_dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=torch.result_type(x, norm.weight))

    ref = norm(x)
    ref.backward(grad)
    ref_dx, x.grad = x.grad.clone(), None
    out = RMSNorm.apply(x, norm.weight, 1e-6)
    out.backward(grad)

    assert out.dtype == ref.dtype
    tol = {torch.float32: 1e-5, torch.bfloat16: 2e-2, torch.float16: 2e-3}[x_dtype]
    torch.testing.assert_close(out, ref, rtol=tol, atol=tol)
    torch.testing.assert_close(x.grad, ref_dx, rtol=tol, atol=tol)

"""Without the padding mask, real tokens of a right-padded batch get the same attention output."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_causal_without_mask_matches_padded(dtype):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    from solution.causal_sdpa import right_padded

    torch.manual_seed(0)
    B, HQ, HK, T, D = 3, 12, 2, 64, 128
    lengths = torch.tensor([64, 40, 7])
    mask2d = (torch.arange(T)[None] < lengths[:, None]).long().cuda()
    assert right_padded(mask2d) and not right_padded(mask2d.flip(1))
    q = torch.randn(B, HQ, T, D, device="cuda", dtype=dtype)
    k = torch.randn(B, HK, T, D, device="cuda", dtype=dtype)
    v = torch.randn(B, HK, T, D, device="cuda", dtype=dtype)
    module = torch.nn.Module()
    module.num_key_value_groups, module.is_causal, module.training = HQ // HK, True, True
    causal = torch.ones(T, T, dtype=torch.bool, device="cuda").tril()
    mask4d = causal[None, None] & mask2d.bool()[:, None, None, :]
    ref, _ = sdpa_attention_forward(module, q, k, v, mask4d)
    out, _ = sdpa_attention_forward(module, q, k, v, None)
    real = mask2d.bool()[:, :, None, None].expand_as(ref)  # [B, T, H, D] after the transpose
    tol = 2e-2 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(out[real], ref[real], rtol=tol, atol=tol)

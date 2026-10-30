"""Our NF4 dequant against bitsandbytes, bit for bit."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(8960, 1536), (256, 1536), (300, 64)])
def test_nf4_dequant_matches_bnb(dtype, shape):
    import bitsandbytes.functional as F

    from solution.nf4 import dequant_nf4

    torch.manual_seed(0)
    w = torch.randn(shape, device="cuda", dtype=dtype)
    packed, qs = F.quantize_4bit(w, blocksize=64, compress_statistics=True, quant_type="nf4")
    ref = F.dequantize_4bit(packed, qs)
    ours = dequant_nf4(packed, qs.absmax, qs.state2.code, qs.state2.absmax, qs.offset,
                       qs.blocksize, qs.shape, qs.dtype)
    torch.testing.assert_close(ours, ref, rtol=0, atol=0)

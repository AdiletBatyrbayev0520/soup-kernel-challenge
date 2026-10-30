"""Fused CE against ForCausalLMLoss + TRL's entropy/accuracy + Soup's per-row loss."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2, 37, 151936), (3, 64, 1001)])
def test_fused_ce_matches_pytorch(dtype, shape):
    from soup_cli.trainer.rewind_hf import row_losses, row_losses_from_token_losses
    from transformers.loss.loss_utils import ForCausalLMLoss
    from trl.trainer.utils import entropy_from_logits

    from solution.fused_ce import fused_causal_lm_loss

    torch.manual_seed(0)
    b, t, v = shape
    logits = (torch.randn(shape, device="cuda") * 3).to(dtype).requires_grad_()
    labels = torch.randint(0, v, (b, t), device="cuda")
    labels[:, : t // 3] = -100
    labels[0, -5:] = -100
    logits.data[0, 3, 7] = logits.data[0, 3, 9] = logits.data[0, 3].max() + 1  # argmax tie
    labels[0, 4] = 9
    n = int((labels[:, 1:] != -100).sum()) + 2  # num_items_in_batch need not equal the count

    ref = ForCausalLMLoss(logits, labels, v, num_items_in_batch=n)
    ref.backward(torch.tensor(3.0, device="cuda"))
    ref_grad, logits.grad = logits.grad.clone(), None
    with torch.no_grad():
        ref_entropy = entropy_from_logits(logits.float())
        pred = logits[:, :-1].float().argmax(-1)
        mask = labels[:, 1:] != -100
        ref_correct, ref_labelled = ((pred == labels[:, 1:]) & mask).sum(), mask.sum()
        ref_rows = row_losses(logits.float(), labels)

    work = logits.detach().clone().requires_grad_()
    loss, step = fused_causal_lm_loss(work, labels, n)
    loss.backward(torch.tensor(3.0, device="cuda"))

    torch.testing.assert_close(loss, ref, rtol=1e-5, atol=1e-5)
    tol = 1e-2 if dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(work.grad.float(), ref_grad.float(), rtol=tol, atol=1e-6)
    torch.testing.assert_close(step.token_entropy, ref_entropy, rtol=1e-4, atol=1e-4)
    assert int(step.correct) == int(ref_correct) and int(step.labelled) == int(ref_labelled)
    rows, counts = row_losses_from_token_losses(step.token_loss, labels)
    assert counts == ref_rows[1]
    torch.testing.assert_close(torch.tensor(rows), torch.tensor(ref_rows[0]), rtol=1e-5, atol=1e-5)

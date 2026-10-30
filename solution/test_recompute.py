"""Layer recompute gives the same loss and LoRA gradients, LoRA dropout included."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_recompute_matches_plain(monkeypatch):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from transformers.models.qwen2 import modeling_qwen2

    from solution import machines
    from solution.recompute import patch_layer_recompute

    config = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2)
    torch.manual_seed(0)
    model = get_peft_model(Qwen2ForCausalLM(config).cuda(),
                           LoraConfig(r=4, lora_dropout=0.5, target_modules=["q_proj", "v_proj"]))
    model.train()
    ids = torch.randint(0, 128, (2, 16), device="cuda")

    def grads():
        model.zero_grad()
        torch.manual_seed(1)
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        return loss.detach(), [p.grad.clone() for p in model.parameters() if p.requires_grad]

    ref_loss, ref = grads()
    original = modeling_qwen2.Qwen2DecoderLayer.forward
    monkeypatch.setattr(modeling_qwen2.Qwen2DecoderLayer, "forward", original)
    monkeypatch.setitem(machines.profile(), "recompute_layers", 2)
    patch_layer_recompute()
    loss, ours = grads()
    torch.testing.assert_close(loss, ref_loss)
    for a, b in zip(ours, ref):
        torch.testing.assert_close(a, b)

"""Recompute decoder layers in backward instead of keeping their activations (our own code).

Each wrapped layer keeps only its fp32 input (12 MiB here) for backward and runs its forward
again there, so the bf16 MLP activations, attention inputs and norm/LoRA saves of that layer
(about 160 MiB each at batch 4 x 512) are not alive across the whole step. Soup's
gradient_checkpointing switch stays off (it is part of the task); this is the "recomputation in
your own code" the rules allow, and its time counts. torch.utils.checkpoint restores the RNG
state before the recompute, so LoRA dropout draws the same masks. The number of layers comes
from the machine profile (recompute_layers: 0 = off, 28 = every layer): a time/memory knob.
Idea from Unsloth's gradient checkpointing (unsloth_zoo/gradient_checkpointing.py), which also
offloads the saved input to CPU; at 12 MiB per layer that is not worth a copy here.
"""

import torch
from torch.utils.checkpoint import checkpoint

from .machines import profile


def patch_layer_recompute():
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2DecoderLayer.forward

    def forward(self, hidden_states, *args, **kwargs):
        layer = getattr(getattr(self, "self_attn", None), "layer_idx", None)
        if (layer is None or not self.training or not torch.is_grad_enabled()
                or layer >= profile()["recompute_layers"]):
            return original(self, hidden_states, *args, **kwargs)
        return checkpoint(original, self, hidden_states, *args, use_reentrant=False,
                          preserve_rng_state=True, **kwargs)

    modeling_qwen2.Qwen2DecoderLayer.forward = forward

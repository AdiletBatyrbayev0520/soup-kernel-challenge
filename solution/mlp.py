"""MLP activations in the compute dtype (fp16 on T4, bf16 on Ampere+) instead of fp32.

prepare_model_for_kbit_training keeps the norms in fp32, so the MLP input is fp32 and every 4-bit
linear returns fp32 (bitsandbytes casts x to its compute dtype, multiplies, and casts the result
back to x's dtype). gate, up and silu(gate) * up, all saved for backward, are then fp32: about
5.9 GB over 28 layers. Casting x once to the linear's compute dtype gives the same products (bnb
rounded x to that dtype anyway, and fp16 results upcast to fp32 and back are exact) at half the
memory. Idea from Unsloth's LoRA_MLP (unsloth/kernels/fast_lora.py), which keeps e and g in the
compute dtype.
"""


def patch_mlp_compute_dtype():
    """Wraps whatever Qwen2MLP.forward is (the example SwiGLU patch, when applied first)."""
    from transformers.models.qwen2 import modeling_qwen2

    inner = modeling_qwen2.Qwen2MLP.forward

    def forward(self, x):
        dtype = getattr(getattr(self, "gate_proj", None), "compute_dtype", None)
        if dtype is not None and x.is_cuda and x.dtype != dtype:
            x = x.to(dtype)
        return inner(self, x)

    modeling_qwen2.Qwen2MLP.forward = forward

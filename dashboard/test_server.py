from server import categorize


def test_categorize_real_kernel_names():
    kernels = [
        ("void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x4_tn_align8>", "matmul"),
        ("void kDequantizeBlockwise<__nv_bfloat16, 512, 64, 8, 2>(float*, unsigned char*)", "nf4_dequant"),
        ("fmha_cutlassB_bf16_aligned_64x64_k128_sm80(PyTorchMemEffAttention::AttentionBackwardKernel)", "attention"),
        ("void at::native::cunn_SoftMaxForward<4, float, float, float>", "softmax_ce"),
        ("void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor", "elementwise"),
        ("void at::native::vectorized_elementwise_kernel<4, at::native::converting_copy_kernel_cuda<float>", "cast_copy"),
        ("void at::native::multi_tensor_apply_kernel<TensorListMetadata<4>, FusedAdamMathFunctor>", "optimizer"),
        ("my_kernel", "own_triton"),
        ("example_swiglu_bwd_kernel", "example_triton"),
    ]
    for name, want in kernels:
        cats, _ = categorize([{"name": name, "time_us": 1000.0}], {"my_kernel", "example_swiglu_bwd_kernel"})
        assert cats[want] == 1.0, (name, want, {k: v for k, v in cats.items() if v})

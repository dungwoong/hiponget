#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <iostream>

__global__ void add_kernel(const float* a, const float* b, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = a[idx] + b[idx];
    }
}

at::Tensor hip_add_manual_copy(const at::Tensor& a, const at::Tensor& b) {
    TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32, "Inputs must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "Input sizes must match");

    int N = a.numel();

    // Allocate output CPU tensor
    auto out = at::empty_like(a, at::kCPU);

    // Allocate HIP device memory
    float *d_a, *d_b, *d_out;
    hipMalloc(&d_a, N * sizeof(float));
    hipMalloc(&d_b, N * sizeof(float));
    hipMalloc(&d_out, N * sizeof(float));

    // Copy input data from CPU or device tensor to HIP device memory
    hipMemcpy(d_a, a.data_ptr<float>(), N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_b, b.data_ptr<float>(), N * sizeof(float), hipMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(add_kernel, dim3(numBlocks), dim3(blockSize), 0, 0, d_a, d_b, d_out, N);

    hipDeviceSynchronize();

    // Copy result back to CPU tensor
    hipMemcpy(out.data_ptr<float>(), d_out, N * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_out);

    return out;
}

PYBIND11_MODULE(func, m) {
    m.def("hip_add_manual_copy", &hip_add_manual_copy, "Add two float tensors via manual HIP memory copies");
}

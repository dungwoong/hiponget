#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <iostream>

template <uint BLOCKSIZE>
__global__ void NaiveGemm(float *a, const float *b, float *c, int M, int N, int K)
{
    const int global_row = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int global_col = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
    c += global_row * N + global_col;
    float tmp = 0;

    for (uint i = 0; i < K; ++i) {
        tmp += a[global_row * K + i] * b[i * N + global_col];
    }
    c[0] = tmp;
}

// should be a way to wrap things with tensors, but dw for now
void matmul_launcher(uintptr_t A, uintptr_t B, uintptr_t C, const int M, const int N, const int K) {
    constexpr int BLOCKSIZE = 32;

    dim3 blockdim(BLOCKSIZE * BLOCKSIZE);
    dim3 griddim(M / BLOCKSIZE, N / BLOCKSIZE);
    float* A_ptr = reinterpret_cast<float*>(A);
    float* B_ptr = reinterpret_cast<float*>(B);
    float* C_ptr = reinterpret_cast<float*>(C);

    hipLaunchKernelGGL(NaiveGemm<BLOCKSIZE>,
        griddim, blockdim, 0, 0,
        A_ptr, B_ptr, C_ptr, M, N, K
    );

    hipDeviceSynchronize();
}

uintptr_t move_to_hip(const at::Tensor& x) {
    TORCH_CHECK(!x.is_cuda(), "Input must be on CPU");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Only float32 supported");

    int64_t numel = x.numel();
    float* device_ptr;
    hipMalloc(&device_ptr, numel * sizeof(float));
    hipMemcpy(device_ptr, x.data_ptr<float>(), numel * sizeof(float), hipMemcpyHostToDevice);

    return reinterpret_cast<uintptr_t>(device_ptr);  // send to Python as int
}

void move_to_torch(uintptr_t addr, const at::Tensor& x) {
    TORCH_CHECK(!x.is_cuda(), "Input must be on CPU");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Only float32 supported");

    int64_t numel = x.numel();
    float *data = reinterpret_cast<float*>(addr);
    hipMemcpy(x.data_ptr<float>(), data, numel * sizeof(float), hipMemcpyDeviceToHost);
}

__global__ void double_kernel(float* data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        data[idx] *= 2.0f;
    }
}

void launch_double(uintptr_t addr, int64_t numel) {
    float* data = reinterpret_cast<float*>(addr);
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    hipLaunchKernelGGL(double_kernel, dim3(blocks), dim3(threads), 0, 0, data, numel);
    hipDeviceSynchronize();
}

void free_hip(uintptr_t ptr_addr) {
    void* ptr = reinterpret_cast<void*>(ptr_addr);
    hipFree(ptr);
}

PYBIND11_MODULE(libv1, m) {
    m.def("to_hip", &move_to_hip, "Copy CPU tensor to HIP and return address");
    m.def("to_torch", &move_to_torch, "Copy HIP tensor to CPU");
    m.def("free_hip", &free_hip, "Free HIP memory");
    m.def("double", &launch_double, "Double elements in HIP memory");
    m.def("matmul", &matmul_launcher, "Matmul");
}
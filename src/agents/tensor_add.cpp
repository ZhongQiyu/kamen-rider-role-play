#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA 内核：用于元素乘以2
__global__ void multiply_by_two_kernel(float* x, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        x[index] *= 2;
    }
}

// CUDA 内核：用于加法操作
__global__ void add_kernel(float* __restrict__ a, float* __restrict__ b, float* __restrict__ result, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        result[index] = a[index] + b[index];
    }
}

// 前向 CUDA 函数：用于加法操作
std::vector<torch::Tensor> cuda_add_forward(torch::Tensor a, torch::Tensor b) {
    // 确保 a 和 b 是相同形状
    TORCH_CHECK(a.sizes() == b.sizes(), "Inputs must be the same size");

    auto result = torch::zeros_like(a);  // 创建与输入 a 大小相同的零张量
    const int threads = 1024;
    const int blocks = (a.numel() + threads - 1) / threads;

    // 调用加法 CUDA 核函数
    add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        result.data_ptr<float>(),
        a.numel()
    );

    // 同步 CUDA 操作
    cudaDeviceSynchronize();

    return {result};
}

// 前向 CUDA 函数：用于 multiply by two 操作
torch::Tensor custom_forward_cuda(torch::Tensor input) {
    const int threads = 1024;
    const int blocks = (input.numel() + threads - 1) / threads;

    // 调用 CUDA 内核
    multiply_by_two_kernel<<<blocks, threads>>>(input.data_ptr<float>(), input.numel());

    // 同步 CUDA 操作
    cudaDeviceSynchronize();

    return input;
}

// C++ 接口：绑定加法和 multiply by two 操作
std::vector<torch::Tensor> add_forward(torch::Tensor a, torch::Tensor b) {
    return cuda_add_forward(a, b);
}

// 将所有操作暴露给 Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // 绑定加法操作
    m.def("add_forward", &add_forward, "Custom CUDA addition forward");

    // 绑定 multiply by two 操作
    m.def("custom_forward_cuda", &custom_forward_cuda, "Multiply by two using CUDA");
}

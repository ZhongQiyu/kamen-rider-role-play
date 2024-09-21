#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA forward declarations
std::vector<torch::Tensor> cuda_add_forward(torch::Tensor a, torch::Tensor b);
torch::Tensor custom_forward_cuda(torch::Tensor input);

// CUDA 内核示例
__global__ void multiply_by_two_kernel(float* x, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        x[index] *= 2;
    }
}

// 前向 CUDA 函数：用于加法操作
std::vector<torch::Tensor> cuda_add_forward(torch::Tensor a, torch::Tensor b) {
    // 确保 a 和 b 是相同形状
    TORCH_CHECK(a.sizes() == b.sizes(), "Inputs must be the same size");

    // 返回 a 和 b 的和
    return {a + b};
}

// 前向 CUDA 函数：用于自定义的 multiply by two 操作
torch::Tensor custom_forward_cuda(torch::Tensor input) {
    const int threads = 1024;
    const int blocks = (input.numel() + threads - 1) / threads;

    // 调用 CUDA 内核
    multiply_by_two_kernel<<<blocks, threads>>>(input.data_ptr<float>(), input.numel());
    
    // 同步 CUDA 操作
    cudaDeviceSynchronize();

    return input;
}

// C++ interface (binding) for the custom addition and custom CUDA kernel
std::vector<torch::Tensor> add_forward(torch::Tensor a, torch::Tensor b) {
  return cuda_add_forward(a, b);
}

// Binding the functions to PyTorch
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // 绑定加法操作
    m.def("add_forward", &add_forward, "Custom CUDA addition forward");

    // 绑定 multiply_by_two 操作
    m.def("custom_forward_cuda", &custom_forward_cuda, "Multiply by two using CUDA");
}

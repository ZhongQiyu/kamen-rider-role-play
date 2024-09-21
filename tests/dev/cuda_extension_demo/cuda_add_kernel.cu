#include <torch/extension.h>
#include <vector>

// CUDA 核函数：用于加法运算
__global__ void add_kernel(float* __restrict__ a, float* __restrict__ b, float* __restrict__ result, const int size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        result[index] = a[index] + b[index];
    }
}

// CUDA 前向操作：执行张量加法
std::vector<torch::Tensor> cuda_add_forward(torch::Tensor a, torch::Tensor b) {
    auto result = torch::zeros_like(a);  // 创建与输入 a 大小相同的零张量

    const int threads = 1024;
    const int blocks = (a.numel() + threads - 1) / threads;

    // 启动 CUDA 内核执行加法操作
    add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        result.data_ptr<float>(),
        a.numel()
    );

    return {result};
}

// 定义一个简单的前向操作
torch::Tensor custom_forward(torch::Tensor input) {
    return input * 2;  // 简单的前向操作：输入张量乘以 2
}

// 定义一个简单的反向操作
torch::Tensor custom_backward(torch::Tensor grad_output) {
    return grad_output * 2;  // 简单的反向操作：输入梯度乘以 2
}

// 将操作暴露给 Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cuda_add_forward", &cuda_add_forward, "CUDA Add forward (CUDA)");
    m.def("forward", &custom_forward, "Custom forward (CUDA)");
    m.def("backward", &custom_backward, "Custom backward (CUDA)");
}

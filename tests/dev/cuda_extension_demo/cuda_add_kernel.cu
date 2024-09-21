
#include <torch/extension.h>

__global__ void add_kernel(float* __restrict__ a, float* __restrict__ b, float* __restrict__ result, const int size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        result[index] = a[index] + b[index];
    }
}

std::vector<torch::Tensor> cuda_add_forward(torch::Tensor a, torch::Tensor b) {
    auto result = torch::zeros_like(a);

    const int threads = 1024;
    const int blocks = (a.numel() + threads - 1) / threads;

    add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        result.data_ptr<float>(),
        a.numel()
    );

    return {result};
}

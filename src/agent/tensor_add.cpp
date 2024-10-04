#include <httplib.h>
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <algorithm>

// CUDA kernel for element-wise multiply by two
__global__ void multiply_by_two_kernel(float* x, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        x[index] *= 2;
    }
}

// CUDA kernel for addition
__global__ void add_kernel(float* __restrict__ a, float* __restrict__ b, float* __restrict__ result, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        result[index] = a[index] + b[index];
    }
}

// Forward CUDA function for addition
std::vector<torch::Tensor> cuda_add_forward(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.sizes() == b.sizes(), "Inputs must be the same size");
    auto result = torch::zeros_like(a); // Create a zero tensor of the same size as input a
    const int threads = 1024;
    const int blocks = (a.numel() + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), result.data_ptr<float>(), a.numel());
    cudaDeviceSynchronize();
    return {result};
}

// Forward CUDA function for multiply by two operation
torch::Tensor custom_forward_cuda(torch::Tensor input) {
    const int threads = 1024;
    const int blocks = (input.numel() + threads - 1) / threads;
    multiply_by_two_kernel<<<blocks, threads>>>(input.data_ptr<float>(), input.numel());
    cudaDeviceSynchronize();
    return input;
}

// TaskEvaluator class to handle task complexities and relatedness calculations
class TaskEvaluator {
public:
    std::vector<float> complexities;
    std::vector<std::vector<float>> relatedness_matrix;

    TaskEvaluator(const std::vector<float>& complexities, const std::vector<std::vector<float>>& relatedness_matrix)
        : complexities(complexities), relatedness_matrix(relatedness_matrix) {}

    float calculate_average_relatedness(int task_index) {
        float sum = std::accumulate(relatedness_matrix[task_index].begin(), relatedness_matrix[task_index].end(), 0.0f);
        return sum / relatedness_matrix[task_index].size();
    }

    std::vector<int> identify_complex_tasks(float complexity_threshold) {
        std::vector<int> indices;
        for (int i = 0; i < complexities.size(); ++i) {
            if (complexities[i] > complexity_threshold) {
                indices.push_back(i);
            }
        }
        return indices;
    }
};

void handle_cuda_operations(const httplib::Request &req, httplib::Response &res) {
    if (req.has_param("operation") && req.get_param_value("operation") == "multiply_by_two") {
        auto input = torch::ones({10}, torch::dtype(torch::kFloat32));
        auto result = custom_forward_cuda(input);
        std::stringstream ss;
        ss << "Result of CUDA multiply by two: ";
        for (int i = 0; i < result.size(0); ++i) {
            ss << result[i].item<float>() << " ";
        }
        res.set_content(ss.str(), "text/plain");
    }
}

int main() {
    httplib::Server svr;

    svr.Get("/cuda", handle_cuda_operations);

    std::cout << "Server is running at http://localhost:8080" << std::endl;
    svr.listen("localhost", 8080);

    return 0;
}

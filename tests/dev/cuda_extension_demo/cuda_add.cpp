
#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
std::vector<torch::Tensor> cuda_add_forward(torch::Tensor a, torch::Tensor b);

// C++ interface (binding)
std::vector<torch::Tensor> add_forward(torch::Tensor a, torch::Tensor b) {
  return cuda_add_forward(a, b);
}

// Binding the function to PyTorch
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add_forward", &add_forward, "Custom CUDA addition forward");
}

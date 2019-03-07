#include <torch/torch.h>

#include <array>

// CUDA forward declarations

at::Tensor unfold_dot_cuda_forward(
    at::Tensor query,           // (batch, head, time1, feat)
    at::Tensor key,             // (batch, head, time2, feat)
    int64_t restrict
    );

std::array<at::Tensor, 2> unfold_dot_cuda_backward(
    at::Tensor dret,            // (batch, head, time1, restrict)
    at::Tensor query,           // (batch, head, time1, feat)
    at::Tensor key              // (batch, head, time2, feat)
    );

at::Tensor unfold_matmul_cuda_forward(
    at::Tensor score,           // (batch, head, time, restrict)
    at::Tensor value            // (batch, head, time, feat)
    );

std::array<at::Tensor, 2> unfold_matmul_cuda_backward(
    at::Tensor dret,            // (batch, head, time, restrict)
    at::Tensor score,           // (batch, head, time, restrict)
    at::Tensor value            // (batch, head, time, feat)
    );


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("unfold_dot_cuda_forward", &unfold_dot_cuda_forward, "UnfoldDot forward (CUDA)");
  m.def("unfold_dot_cuda_backward", &unfold_dot_cuda_backward, "UnfoldDot backward (CUDA)");
  m.def("unfold_matmul_cuda_forward", &unfold_matmul_cuda_forward, "UnfoldMatmul forward (CUDA)");
  m.def("unfold_matmul_cuda_backward", &unfold_matmul_cuda_backward, "UnfoldMatmul backward (CUDA)");
}

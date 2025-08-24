#include <torch/extension.h>

at::Tensor xor_popcount_launcher(const at::Tensor& A, const at::Tensor& B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("xor_popcount", &xor_popcount_launcher, "popcount(xor(a,b))");
}

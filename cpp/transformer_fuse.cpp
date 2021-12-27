#include <torch/extension.h>

#include <vector>

using Tensor = torch::Tensor;

Tensor transformer_fuse_forward(Tensor input, Tensor weights, Tensor bias) {
    auto output = torch::addmm(bias, input, weights.transpose(0, 1));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &transformer_fuse_forward, "transformer_fuse_forward");
}

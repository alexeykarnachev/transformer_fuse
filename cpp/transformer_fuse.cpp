#include <torch/extension.h>

#include <vector>

using Tensor = torch::Tensor;

const size_t NUM_ATTENTION_HEAD = 12;
const size_t ATTENTION_HEAD_SIZE = 64;
const size_t HIDDEN_SIZE = 768;
const float ATTENTION_DEVISOR = 8.0;

Tensor bert_self_attention(
    Tensor hidden_states,
    Tensor query_w,
    Tensor query_b,
    Tensor key_w,
    Tensor key_b,
    Tensor value_w,
    Tensor value_b) {
    auto bs = hidden_states.size(0);
    auto seq_len = hidden_states.size(1);

    auto query = torch::matmul(hidden_states, query_w.transpose(0, 1)) + query_b;
    auto key = torch::matmul(hidden_states, key_w.transpose(0, 1)) + key_b;
    auto value = torch::matmul(hidden_states, value_w.transpose(0, 1)) + value_b;

    query = query.view({bs, seq_len, NUM_ATTENTION_HEAD, ATTENTION_HEAD_SIZE}).permute({0, 2, 1, 3});
    key = key.view({bs, seq_len, NUM_ATTENTION_HEAD, ATTENTION_HEAD_SIZE}).permute({0, 2, 1, 3});
    value = value.view({bs, seq_len, NUM_ATTENTION_HEAD, ATTENTION_HEAD_SIZE}).permute({0, 2, 1, 3});

    auto attention_scores = query.matmul(key.transpose(3, 2)) / ATTENTION_DEVISOR;
    auto attention_probs = torch::softmax(attention_scores, 3);

    auto context = attention_probs.matmul(value)
                       .permute({0, 2, 1, 3})
                       .contiguous()
                       .view({bs, seq_len, HIDDEN_SIZE});
    return context;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &bert_self_attention, "bert_self_attention");
}

import datetime

import numpy as np
import torch
import tqdm
import transformer_fuse_cpp
from torch import nn
from transformers.models.bert.modeling_bert import BertConfig, BertSelfAttention

torch.manual_seed(42)

HIDDEN_SIZE = 768


class MyBertSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.query_weight = nn.Parameter(
            torch.zeros((HIDDEN_SIZE, HIDDEN_SIZE), dtype=torch.float32),
            requires_grad=False,
        )
        self.key_weight = nn.Parameter(
            torch.zeros((HIDDEN_SIZE, HIDDEN_SIZE), dtype=torch.float32),
            requires_grad=False,
        )
        self.value_weight = nn.Parameter(
            torch.zeros((HIDDEN_SIZE, HIDDEN_SIZE), dtype=torch.float32),
            requires_grad=False,
        )
        self.query_bias = nn.Parameter(
            torch.zeros((HIDDEN_SIZE, ), dtype=torch.float32),
            requires_grad=False,
        )
        self.key_bias = nn.Parameter(
            torch.zeros((HIDDEN_SIZE, ), dtype=torch.float32),
            requires_grad=False,
        )
        self.value_bias = nn.Parameter(
            torch.zeros((HIDDEN_SIZE, ), dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, hidden_states):
        return transformer_fuse_cpp.forward(
            hidden_states,
            self.query_weight,
            self.query_bias,
            self.key_weight,
            self.key_bias,
            self.value_weight,
            self.value_bias,
        )


def benchmark(model, input, n):
    results = []
    for _ in tqdm.trange(n):
        start = datetime.datetime.now()
        model(input)
        end = datetime.datetime.now()
        res = (end - start).total_seconds() / n
        res *= 1000
        results.append(res)

    return np.median(results)


if __name__ == '__main__':
    seq_len = 128
    batch_size = 256
    n_reps = 10000
    device = 'cuda'

    with torch.no_grad():
        config = BertConfig.from_pretrained('bert-base-uncased')
        model_b = BertSelfAttention(config).to(device).eval()
        model_a = MyBertSelfAttention().to(device)
        state_dict = {key.replace('.', '_'): val for key, val in model_b.state_dict().items()}
        model_a.load_state_dict(state_dict)
        hidden_sates = torch.randn((batch_size, seq_len, HIDDEN_SIZE), device=device)

        out_a = model_a(hidden_sates)
        (out_b, ) = model_b(hidden_sates)
        assert (torch.isclose(out_a, out_b).all())
        print(f'Torch model: {benchmark(model_b, hidden_sates, n_reps):.4f}')
        print(f'Custom model: {benchmark(model_a, hidden_sates, n_reps):.4f}')

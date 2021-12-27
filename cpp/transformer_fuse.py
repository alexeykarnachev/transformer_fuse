import datetime

import torch
import transformer_fuse_cpp
from torch import nn
from torch.autograd import Function

torch.manual_seed(42)


class TransformerFuseFunction(Function):
    @staticmethod
    def forward(ctx, input, weights, bias):
        output = transformer_fuse_cpp.forward(input, weights, bias)
        return output


class TransformerFuse(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros((out_features, in_features), dtype=torch.float32),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.zeros((out_features, ), dtype=torch.float32),
            requires_grad=False,
        )
        self.reset_parameters()

    @property
    def device(self):
        return self.weight.data.device

    def reset_parameters(self):
        for weight in self.parameters():
            weight.data.uniform_(-1, 1)

    def forward(self, input):
        return TransformerFuseFunction.apply(input, self.weight, self.bias)


def benchmark(model, input, n):
    start = datetime.datetime.now()
    for _ in range(n):
        model(input)
    end = datetime.datetime.now()
    res = (end - start).total_seconds() / n
    res *= 1000
    return res


if __name__ == '__main__':
    in_features = 2048
    out_features = 2048
    batch_size = 128
    n_reps = 100000
    device = 'cuda'
    with torch.no_grad():
        input = torch.randn((batch_size, in_features), device=device)

        model_a = TransformerFuse(in_features=in_features, out_features=out_features).to(device)
        model_b = nn.Linear(in_features, out_features).to(device)
        model_b.load_state_dict(model_a.state_dict())
        model_a(input)
        model_b(input)

        print(f'Custom model: {benchmark(model_a, input, n_reps):.4f}')
        print(f'Torch model: {benchmark(model_b, input, n_reps):.4f}')

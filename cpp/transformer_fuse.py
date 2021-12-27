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
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.weights = nn.Parameter(torch.zeros((1, n_features), dtype=torch.float32), requires_grad=False)
        self.bias = nn.Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=False)
        self.reset_parameters()

    @property
    def device(self):
        return self.weights.data.device

    def reset_parameters(self):
        for weight in self.parameters():
            weight.data.uniform_(-1, 1)

    def forward(self, input):
        return TransformerFuseFunction.apply(input, self.weights, self.bias)


if __name__ == '__main__':
    n_features = 1024
    batch_size = 64
    model = TransformerFuse(n_features=n_features).to('cuda')
    input = torch.randn((batch_size, n_features), device=model.device)
    output = model(input)
    print(output)

import numpy as np

from mytorch import Tensor
from mytorch.layer import Layer
from mytorch.util import initializer


class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1),
                 need_bias: bool = False, mode="xavier") -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.need_bias = need_bias
        self.weight: Tensor = None
        self.bias: Tensor = None
        self.initialize_mode = mode

        self.initialize()

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"
        size = x.shape[0]
        dim1 = int((x.shape[2] + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0]) + 1
        dim2 = int((x.shape[3] + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1]) + 1

        output = np.zeros((size, self.out_channels, dim1, dim2))
        padding_x = Tensor(np.pad(x.data, ((0,), (0,), (self.padding[0],), (self.padding[1],))))

        for f in range(self.out_channels):
            for i in range(dim1):
                start_i = i * self.stride[0]
                end_i = start_i + self.kernel_size[0]

                for j in range(dim2):
                    start_j = j * self.stride[1]
                    end_j = start_j + self.kernel_size[1]

                    value = 0
                    for c in range(self.in_channels):
                        value += (padding_x[:, c, start_i:end_i, start_j:end_j] * self.weight[f, c]).sum_in_batch()

                    if self.need_bias:
                        value += self.bias[f]

                    output[:, f, i, j] = value.data
        return output

    def initialize(self):
        self.weight = Tensor(
            data=initializer((self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]),
                             self.initialize_mode),
            requires_grad=True
        )
        self.bias = Tensor(
            data=initializer((self.out_channels, 1), "zero"),
            requires_grad=True
        )

    def zero_grad(self):
        "TODO: implement zero grad"
        self.weight.zero_grad()
        if self.need_bias:
            self.bias.zero_grad()

    def parameters(self):
        "TODO: return weights and bias"
        if self.need_bias:
            return [self.weight, self.bias]

        return [self.weight]

    def __str__(self) -> str:
        return "conv 2d - total params: {} - kernel: {}, stride: {}, padding: {}".format(
            self.kernel_size[0] * self.kernel_size[1],
            self.kernel_size,
            self.stride, self.padding)

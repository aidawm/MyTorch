import numpy as np

from mytorch import Tensor
from mytorch.layer import Layer


class AvgPool2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)) -> None:
        super()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"
        assert x.shape[1] == self.out_channels
        size = x.shape[0]

        dim1 = int((x.shape[2] + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0]) + 1
        dim2 = int((x.shape[3] + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1]) + 1

        output = np.zeros((size, self.out_channels, dim1, dim2))
        padding_x = np.pad(x.data, ((0,), (0,), (self.padding[0],), (self.padding[1],)))
        for i in range(dim1):
            start_i = i * self.stride[0]
            end_i = start_i + self.kernel_size[0]

            for j in range(dim2):
                start_j = j * self.stride[1]
                end_j = start_j + self.kernel_size[1]

                for c in range(self.in_channels):
                    value = np.average(padding_x[:, c, start_i:end_i, start_j:end_j].data, axis=(1, 2))
                    output[:, c, i, j] = value

        return Tensor(output)

    def __str__(self) -> str:
        return "avg pool 2d - kernel: {}, stride: {}, padding: {}".format(self.kernel_size, self.stride, self.padding)

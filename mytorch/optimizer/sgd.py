from typing import List

from mytorch.layer import Layer
from mytorch.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, layers: List[Layer], learning_rate=0.1):
        super().__init__(layers)
        self.learning_rate = learning_rate

    def step(self):
        for layer in self.layers:
            params = layer.parameters()
            layer.weight = layer.weight - self.learning_rate * params[0].grad
            if layer.need_bias:
                layer.bias = layer.bias - self.learning_rate * params[1].grad

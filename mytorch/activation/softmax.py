import numpy as np

from mytorch import Tensor


def softmax(x: Tensor) -> Tensor:
    e = x.exp()
    s = e @ (np.ones((e.shape[-1], 1)))
    return e * (s ** -1)

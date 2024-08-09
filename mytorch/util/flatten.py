from mytorch import Tensor


def flatten(x: Tensor) -> Tensor:
    """
    TODO: implement flatten.
    this methods transforms a n dimensional array into a flat array
    hint: use numpy flatten
    """
    # print(x.data.shape)
    data = x.data.reshape(x.data.shape[0], -1)
    requires_grad = x.requires_grad
    depends_on = x.depends_on
    return Tensor(data=data, requires_grad=requires_grad, depends_on=depends_on)

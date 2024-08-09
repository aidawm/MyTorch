from mytorch import Tensor


def CategoricalCrossEntropy(preds: Tensor, label: Tensor):
    return -(label * preds.log()).sum()

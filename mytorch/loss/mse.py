from mytorch import Tensor
import numpy as np

def MeanSquaredError(preds: Tensor, actual: Tensor):
    "TODO: implement Mean Squared Error loss"
    # diff = preds - actual
    # sum_diff = (diff * diff).sum() 
    # sum_diff.data = sum_diff.data / diff.data.size
    # return sum_diff

    error = preds - actual
    error2 = error**2
    mse = error2
    size = Tensor(np.array([error2.data.size],dtype=np.float64))
    size = size**-1
    return mse * size
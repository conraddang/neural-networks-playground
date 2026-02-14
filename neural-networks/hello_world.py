import torch
import numpy as np

numpy_array = np.ndarray([1, 2, 3])
print("Numpy array:", numpy_array)

tensor_array = torch.from_numpy(numpy_array)
print("Tensor array:", tensor_array)
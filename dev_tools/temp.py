import os
import platform
import numpy as np

#
# def to_int(x, frac=5):
#     return int(x * 2**frac)
#
# def to_float(x, frac=5):
#     return x / 2**frac
#
# print(to_int(-0.001953125, frac=9))
# print(to_int(to_int(-0.001953125, frac=9), frac=9))
# print(to_float(-1, frac=9))

# Example to read and reshape the data
# data = np.fromfile("../hw_outputs/test_image.data", dtype=np.float32)  # dtype should match the original
# # data = data.reshape([1,1000])  # Replace `original_shape` with the actual shape
# print(data)

import torch
import torch.nn.functional as F

# Input tensor with batch size 1, 1 channel, and size 5x5
input_tensor = torch.arange(1., 51.).view(1, 2, 5, 5)

# Apply unfold with kernel size 3x3, stride 1
unfolded = F.unfold(input_tensor, kernel_size=(3, 3), stride=1)

print(input_tensor.shape)
print(input_tensor)
print(unfolded.shape)
print(unfolded)
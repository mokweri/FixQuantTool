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
data = np.fromfile("../hw_outputs/test_image.data", dtype=np.float32)  # dtype should match the original
# data = data.reshape([1,1000])  # Replace `original_shape` with the actual shape
print(data)
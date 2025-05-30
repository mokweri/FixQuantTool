import os
import platform
import numpy as np


def to_int(x, frac=5):
    return int(x * 2**frac)

def to_float(x, frac=5):
    return x / 2**frac


# print(to_float(-140, frac=8))
# print(to_float(18, frac=6))
# print(to_float(-34, frac=7))
# print(to_int(-0.546875+0.28125, frac=7))

# Example to read and reshape the data
# data = np.fromfile("../hw_outputs/test_image.data", dtype=np.float32)  # dtype should match the original
# # data = data.reshape([1,1000])  # Replace `original_shape` with the actual shape
# print(data)

# import torch
# import torch.nn.functional as F
#
# # Input tensor with batch size 1, 1 channel, and size 5x5
# input_tensor = torch.arange(1., 51.).view(1, 2, 5, 5)
#
# # Apply unfold with kernel size 3x3, stride 1
# unfolded = F.unfold(input_tensor, kernel_size=(3, 3), stride=1)
#
# print(input_tensor.shape)
# print(input_tensor)
# print(unfolded.shape)
# print(unfolded)

qconfig = {'x': {'out': 5}, 'conv1': {'weight': 5, 'bias': 2, 'in': [5], 'out': 1}, 'maxpool': {'in': [1], 'out': 1}, 'layer1_0_conv1': {'weight': 7, 'bias': 3, 'in': [1], 'out': 3}, 'layer1_0_conv2': {'weight': 8, 'bias': 3, 'in': [3], 'out': 2}, 'layer1_0_conv3': {'weight': 7, 'bias': 3, 'in': [2], 'out': 3}, 'layer1_0_downsample_0': {'weight': 8, 'bias': 3, 'in': [1], 'out': 2}, 'add': {'in': [3, 2], 'out': 4}, 'layer1_1_conv1': {'weight': 8, 'bias': 3, 'in': [4], 'out': 3}, 'layer1_1_conv2': {'weight': 8, 'bias': 5, 'in': [3], 'out': 3}, 'layer1_1_conv3': {'weight': 6, 'bias': 5, 'in': [3], 'out': 4}, 'add_1': {'in': [4, 4], 'out': 4}, 'layer1_2_conv1': {'weight': 8, 'bias': 5, 'in': [4], 'out': 4}, 'layer1_2_conv2': {'weight': 8, 'bias': 4, 'in': [4], 'out': 3}, 'layer1_2_conv3': {'weight': 6, 'bias': 3, 'in': [3], 'out': 4}, 'add_2': {'in': [4, 4], 'out': 4}, 'layer2_0_conv1': {'weight': 7, 'bias': 4, 'in': [4], 'out': 3}, 'layer2_0_conv2': {'weight': 9, 'bias': 4, 'in': [3], 'out': 3}, 'layer2_0_conv3': {'weight': 7, 'bias': 4, 'in': [3], 'out': 4}, 'layer2_0_downsample_0': {'weight': 8, 'bias': 4, 'in': [4], 'out': 3}, 'add_3': {'in': [4, 3], 'out': 3}, 'layer2_1_conv1': {'weight': 8, 'bias': 5, 'in': [3], 'out': 4}, 'layer2_1_conv2': {'weight': 8, 'bias': 3, 'in': [4], 'out': 5}, 'layer2_1_conv3': {'weight': 7, 'bias': 4, 'in': [5], 'out': 4}, 'add_4': {'in': [4, 3], 'out': 3}, 'layer2_2_conv1': {'weight': 8, 'bias': 6, 'in': [3], 'out': 4}, 'layer2_2_conv2': {'weight': 8, 'bias': 4, 'in': [4], 'out': 3}, 'layer2_2_conv3': {'weight': 6, 'bias': 4, 'in': [3], 'out': 4}, 'add_5': {'in': [4, 3], 'out': 4}, 'layer2_3_conv1': {'weight': 8, 'bias': 5, 'in': [4], 'out': 4}, 'layer2_3_conv2': {'weight': 7, 'bias': 4, 'in': [4], 'out': 3}, 'layer2_3_conv3': {'weight': 7, 'bias': 3, 'in': [3], 'out': 4}, 'add_6': {'in': [4, 4], 'out': 3}, 'layer3_0_conv1': {'weight': 8, 'bias': 4, 'in': [3], 'out': 3}, 'layer3_0_conv2': {'weight': 9, 'bias': 5, 'in': [3], 'out': 4}, 'layer3_0_conv3': {'weight': 7, 'bias': 3, 'in': [4], 'out': 3}, 'layer3_0_downsample_0': {'weight': 8, 'bias': 4, 'in': [3], 'out': 4}, 'add_7': {'in': [3, 4], 'out': 3}, 'layer3_1_conv1': {'weight': 9, 'bias': 5, 'in': [3], 'out': 4}, 'layer3_1_conv2': {'weight': 8, 'bias': 5, 'in': [4], 'out': 4}, 'layer3_1_conv3': {'weight': 7, 'bias': 5, 'in': [4], 'out': 4}, 'add_8': {'in': [4, 3], 'out': 4}, 'layer3_2_conv1': {'weight': 9, 'bias': 5, 'in': [4], 'out': 4}, 'layer3_2_conv2': {'weight': 8, 'bias': 4, 'in': [4], 'out': 4}, 'layer3_2_conv3': {'weight': 7, 'bias': 5, 'in': [4], 'out': 4}, 'add_9': {'in': [4, 4], 'out': 4}, 'layer3_3_conv1': {'weight': 8, 'bias': 5, 'in': [4], 'out': 4}, 'layer3_3_conv2': {'weight': 8, 'bias': 4, 'in': [4], 'out': 4}, 'layer3_3_conv3': {'weight': 7, 'bias': 4, 'in': [4], 'out': 4}, 'add_10': {'in': [4, 4], 'out': 4}, 'layer3_4_conv1': {'weight': 9, 'bias': 5, 'in': [4], 'out': 4}, 'layer3_4_conv2': {'weight': 8, 'bias': 4, 'in': [4], 'out': 4}, 'layer3_4_conv3': {'weight': 7, 'bias': 4, 'in': [4], 'out': 4}, 'add_11': {'in': [4, 4], 'out': 3}, 'layer3_5_conv1': {'weight': 9, 'bias': 5, 'in': [3], 'out': 4}, 'layer3_5_conv2': {'weight': 8, 'bias': 4, 'in': [4], 'out': 3}, 'layer3_5_conv3': {'weight': 7, 'bias': 4, 'in': [3], 'out': 4}, 'add_12': {'in': [4, 3], 'out': 3}, 'layer4_0_conv1': {'weight': 8, 'bias': 5, 'in': [3], 'out': 3}, 'layer4_0_conv2': {'weight': 9, 'bias': 4, 'in': [3], 'out': 3}, 'layer4_0_conv3': {'weight': 7, 'bias': 3, 'in': [3], 'out': 2}, 'layer4_0_downsample_0': {'weight': 8, 'bias': 4, 'in': [3], 'out': 3}, 'add_13': {'in': [2, 3], 'out': 1}, 'layer4_1_conv1': {'weight': 9, 'bias': 4, 'in': [1], 'out': 4}, 'layer4_1_conv2': {'weight': 9, 'bias': 5, 'in': [4], 'out': 4}, 'layer4_1_conv3': {'weight': 7, 'bias': 3, 'in': [4], 'out': 3}, 'add_14': {'in': [3, 1], 'out': 2}, 'layer4_2_conv1': {'weight': 9, 'bias': 4, 'in': [2], 'out': 6}, 'layer4_2_conv2': {'weight': 10, 'bias': 3, 'in': [6], 'out': 3}, 'layer4_2_conv3': {'weight': 6, 'bias': 4, 'in': [3], 'out': 3}, 'add_15': {'in': [3, 2], 'out': 1}, 'avgpool': {'in': [1], 'out': 3}, 'fc': {'weight': 6, 'bias': 7, 'in': [3], 'out': 3}}


# Define whether dealing with VGG or ResNet
is_vgg = False

# Generate precision config
precision_config = []
addition_precision_config = []

for key, values in qconfig.items():
    if 'features' in key or 'classifier' in key or 'conv' in key or 'fc' in key:
        precision_entry = [
            values.get('weight', 0),
            values.get('bias', 0),
            values['in'][0],
            values['out']
        ]
        precision_config.append(precision_entry)
        if not is_vgg and 'add' not in key:
            addition_precision_config.append([0, 0, 0, 0])

    # Consider downsample add case for ResNet - adjust the logic as needed for other models
    elif 'add' in key:
        addition_entry = [
            1,
            values['in'][0],
            values['in'][1],
            values['out']
        ]
        addition_precision_config.append(addition_entry)

# Write to a header file
header_content = """\
#ifndef CONFIG_H
#define CONFIG_H

static signed char precision_config[][4] = {
"""

for config in precision_config:
    header_content += f"    {{{', '.join(map(str, config))}}},\n"

header_content += """\
};

static char addition_precision_config[][4] = {
"""

# Only include addition layers if not handling VGG
if not is_vgg:
    for config in addition_precision_config:
        header_content += f"    {{{', '.join(map(str, config))}}},\n"

header_content += """\
};

#endif // CONFIG_H
"""

with open('config.h', 'w') as file:
    file.write(header_content)


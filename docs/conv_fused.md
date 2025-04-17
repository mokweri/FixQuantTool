# conv_fused 

This Python project contains implementations for different classes and methods 
that are mostly revolved around Convolutional layers and Batch Normalization Layers in neural networks.

## Main Components
class - `_FusedModule`

class - `_ConvBnNd`

class - `QuantizedConvBatchNorm2d` 

## Description
This module contains different methods and classes for working with and manipulating 
Convolutional and Batch Normalization layers.

The `_FusedModule` 
class doesn't contain much except that it subclasses `torch.nn.Sequential` which is a container of `Module`s that can be stacked together and run at the same time sequentially.


The `_ConvBnNd` 
is used for fusing Convolutional and Batch Normalization layers.It has methods to handle the batch normalization stats, application of the batch normalization to the Convolutional layer among others.

`QuantizedConvBatchNorm2d` is a specialized class that deals with the fusion of a 2-dimensional convolution layer and a 2-dimensional batch normalization layer where the weights are quantized for lower memory consumption and faster processing.


There are also utility functions in the module.

- `update_bn_stats(mod)`
for updating the batch normalization stats on `mod` if it is type `_FusedModule`.
- `freeze_bn_stats(mod)`, you can
freeze the batch normalization statistics.
- `fuse_conv_bn(mod)`, fuse the batch normalization with a convolution layer.
- And finally, `clear_non_native_bias(mod)`, clear bias that was created outside of the original
`torch.nn`
creation process if `mod` is of type '_FusedModule' or '`QuantizedConvBatchNorm2d`.

## Usage

Import the module and create an instance of the class .
`QuantizedConvBatchNorm2d`

This class is used to create a convolutional layer that can have batch normalization fused with it, with weights that can be quantized.The batch normalization can be turned on and off through the '.train(mode=True/False)' method inherited from `torch.nn.Module`.

## Conclusion


The purpose of this project is to assist in handling convolution layers and batch normalization
layers of a neural network, with functionalities easy enough to comprehend and use.

"""
MIT License

Autoencoder-based Attribute Noise Handling Method for Medical Data

Copyright (c) 2022 Thomas RANVIER

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch.nn as nn


# Create possibility to dynamically add layers to sequential
def add_module(self, module):
    self.add_module(str(len(self) + 1), module)

nn.Module.add = add_module

def act(act_fun='LeakyReLU'):
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
    else:
        return act_fun(inplace=False)


def conv_1d(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, bias: bool = True,
            pad: str = 'zero', downsample: str = 'stride'):
    """
    This method returns a 1 dimensional convolutional layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of kernels.
        stride (int): Size of the stride. Defaults to 1.
        bias (bool): Either to set biases or not. Defaults to True.
        pad (str): Type of padding. Defaults to 'zero'. Choose from {'zero', 'reflection'}.
        downsample (str): Type of downsampling. Defaults to 'stride'. Choose from {'stride', 'avg', 'max'}.

    Returns:
        A torch.nn.Sequential object implementing a 1D conv layer composed of a padder, a convolver and a downsampler.
    """
    # Create the downsampler, either using natural stride sizes, or an AvgPool1d or a MaxPool1d downsampler.
    downsampler = None
    if stride != 1 and downsample != 'stride':
        if downsample == 'avg':
            downsampler = nn.AvgPool1d(stride, stride)
        elif downsample == 'max':
            downsampler = nn.MaxPool1d(stride, stride)
        stride = 1

    # Create the padder, either standard 0 padding or using ReflectionPad1d.
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad1d(to_pad)
        to_pad = 0

    # Create the conv layer
    convolver = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=to_pad, bias=bias)

    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)
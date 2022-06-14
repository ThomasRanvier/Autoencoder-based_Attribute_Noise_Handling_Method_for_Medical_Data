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

import torch
from .common import *


class Skip(nn.Module):
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 channels_skip: int,
                 down_layers: list,
                 need_sigmoid: bool = True,
                 need_bias: bool = True,
                 pad: str = 'zero',
                 downsample_mode: str = 'stride',
                 upsample_mode: str = 'nearest',
                 act_fun: object = 'LeakyReLU',
                 need1x1_up: bool = True):
        """
        Initialize a generic convolutional architecture in an auto-encoder manner with skip connections.

        Args:
            input_channels (int): Number of input channels.
            output_channels (int): Number of output channels.
            channels_skip (int): Number of skip connections channels.
            down_layers (list): A list of tuples of the form [('channels', 'filter_size', 'downsample'), ...], where
                                each tuple contains the number of channels, filter size and downsample type of the
                                corresponding layer to add to the model.
            need_sigmoid (bool): If we add a sigmoid after the last layer. Defaults to True.
            need_bias (bool): If we add biases to each conv layer. Defaults to True.
            pad (str): Type of padding. Defaults to 'zero'. Choose from {'zero', 'reflection'}.
            downsample_mode (str): Type of downsampling. Defaults to 'stride'. Choose from {'stride', 'avg', 'max'}.
            upsample_mode (str): Type of upsampling. Defaults to 'nearest'. Choose from {'nearest', 'linear', 'bilinear',
                                 'bicubic', 'trilinear'}.
            act_fun (str or torch activation): The name of the activation function to use. Defaults to 'LeakyReLU'.
            need1x1_up (bool): If we need a conv layer with filter size and stride both at 1 after each up layer.
                               Defaults at True.
        """
        super(Skip, self).__init__()

        if channels_skip <= 0:
            channels_skip = 1

        self._skips = []
        self._down_layers = []
        input_depth = input_channels

        self._skip_init = nn.Sequential()
        self._skip_init.add(conv_1d(input_depth, channels_skip, 1, bias=need_bias, pad=pad))
        self._skip_init.add(nn.BatchNorm1d(channels_skip))
        self._skip_init.add(act(act_fun))
        self._skips.append(self._skip_init)

        for i, (channels, filter_size, downsample) in enumerate(down_layers):
            exec(f'self._down_{i} = nn.Sequential()')
            exec(f'self._down_{i}.add(conv_1d({input_depth}, {channels}, {filter_size}, ' +
                 f'{downsample}, bias={need_bias}, pad=pad, downsample=downsample_mode))')
            exec(f'self._down_{i}.add(nn.BatchNorm1d({channels}))')
            exec(f'self._down_{i}.add(act(act_fun))')
            exec(f'self._down_layers.append(self._down_{i})')

            exec(f'self._skip_{i} = nn.Sequential()')
            exec(f'self._skip_{i}.add(conv_1d({input_depth}, {channels_skip}, 1, ' +
                 f'bias={need_bias}, pad=pad))')
            exec(f'self._skip_{i}.add(nn.BatchNorm1d({channels_skip}))')
            exec(f'self._skip_{i}.add(act(act_fun))')
            exec(f'self._skips.append(self._skip_{i})')

            input_depth = channels

        up_layers = down_layers[::-1]

        self._up_layers = []
        for i, (channels, filter_size, downsample) in enumerate(up_layers):
            first_up = int(i == 0)
            if first_up:
                exec(f'self._first_up = nn.Sequential()')
                exec(
                    f'self._first_up.add(conv_1d({input_depth}, {channels}, {filter_size}, bias={need_bias}, pad=pad))')
                exec(f'self._first_up.add(nn.BatchNorm1d({channels}))')
                exec(f'self._first_up.add(act(act_fun))')
                exec(f'self._first_up.add(nn.Upsample(scale_factor={downsample}, mode=upsample_mode))')
                exec(f'self._up_layers.append(self._first_up)')
            exec(f'self._up_{i} = nn.Sequential()')
            exec(f'self._up_{i}.add(nn.BatchNorm1d({input_depth + channels_skip}))')
            exec(f'self._up_{i}.add(conv_1d({input_depth + channels_skip}, {channels}, ' +
                 f'{filter_size}, bias={need_bias}, pad=pad))')
            exec(f'self._up_{i}.add(nn.BatchNorm1d({channels}))')
            exec(f'self._up_{i}.add(act(act_fun))')
            if need1x1_up:
                exec(f'self._up_{i}.add(conv_1d({channels}, {channels}, 1, ' +
                     f'bias={need_bias}, pad=pad))')
                exec(f'self._up_{i}.add(nn.BatchNorm1d({channels}))')
                exec(f'self._up_{i}.add(act(act_fun))')
            if not i == len(up_layers) - 1:
                exec(f'self._up_{i}.add(nn.Upsample(scale_factor={up_layers[i + 1][2]}, mode=upsample_mode))')
            exec(f'self._up_layers.append(self._up_{i})')
            input_depth = channels

        self._last_conv = nn.Sequential()
        self._last_conv.add(conv_1d(channels + channels_skip, output_channels, 1, bias=need_bias, pad=pad))
        if need_sigmoid:
            self._last_conv.add(nn.Sigmoid())

    def forward(self, x: object):
        """
        Forward propagation through the skip model.

        Args:
            x (torch Tensor): The data to propagate.

        Returns:
            A torch Tensor object, the output from the model.
        """
        skips = []
        skips.append(self._skips[0](x))
        for i, layer in enumerate(self._down_layers):
            skips.append(self._skips[i + 1](x))
            x = layer(x)

        skips = skips[::-1]
        for i, layer in enumerate(self._up_layers):
            x = layer(x)
            x = torch.cat([x, skips[i]], dim=-2)

        x = self._last_conv(x)
        return x
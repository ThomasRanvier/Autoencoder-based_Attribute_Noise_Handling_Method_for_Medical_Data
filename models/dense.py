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


class Dense(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 neurons: list):
        """
        Initialize a dynamic dense model.

        Args:
            input_size (int): Size of the input.
            neurons (list): A list of the sizes of each dense layer (int).
        """
        super(Dense, self).__init__()
        layers = []
        for i, n in enumerate(neurons):
            layers.append(torch.nn.Linear(input_size if i == 0 else neurons[i - 1], n))
            layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
        layers.append(torch.nn.Linear(n, input_size))
        layers.append(torch.nn.Sigmoid())
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x: object):
        """
        Forward propagation through the dense model.

        Args:
            x (torch Tensor): The data to propagate.

        Returns:
            A torch Tensor object, the output from the model.
        """
        for lay in self.layers:
            x = lay(x)
        return x 
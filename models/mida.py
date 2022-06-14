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

from utils import *
from .common import *

dtype = torch.cuda.FloatTensor


class MIDA_model(nn.Module):
    def __init__(self,
                 input_size: int,
                 num_layers: int,
                 theta: int = 7):
        """
        Initialize a MIDA model.

        Args:
            input_size (int): The size of the input.
            num_layers (int): Number of layers in both the encoder and decoder.
            theta (int): Number of additional neuron for each deeper layer.
        """
        super(MIDA_model, self).__init__()

        self._dropout = nn.Dropout(p=.5)
        self._layers = []
        for i in range(num_layers):
            exec(f'self._{i} = nn.Sequential()')
            exec(f'self._{i}.add(nn.Linear({input_size + i * theta}, {input_size + (i + 1) * theta}))')
            exec(f'self._{i}.add(act("LeakyReLU"))')
            exec(f'self._layers.append(self._{i})')
        for i in range(num_layers):
            exec(f'self._{i + num_layers} = nn.Sequential()')
            exec(f'self._{i + num_layers}.add(nn.Linear({input_size + (num_layers - i) * theta}' +
                 f', {input_size + (num_layers - 1 - i) * theta}))')
            exec(f'self._{i + num_layers}.add(act("LeakyReLU"))')
            exec(f'self._layers.append(self._{i + num_layers})')

    def forward(self, x: object):
        """
        Forward propagation through the MIDA model.

        Args:
            x (torch Tensor): The data to propagate.

        Returns:
            A torch Tensor object, the output from the model.
        """
        x = self._dropout(x)
        for lay in self._layers:
            x = lay(x)
        return x


def mida(data_missing_nans: object, num_layers: int, num_epochs: int = 500, nb_batches: int = 10, theta: int = 7):
    """
    Implementation of the MIDA optimization function described by Gondara and Wang.

    Args:
        data_missing_nans (numpy array): The data with missing values replaced as NaNs.
        num_layers (int): Number of layers in both the encoder and decoder of the MIDA model.
        num_epochs (int): Number of epochs to train the model for. Defaults to 500.
        nb_batches (int): Number of batches. Defaults to 10.
        theta (int): Number of additional neuron for each deeper layer. Defaults to 7.

    Returns:
        A numpy array, the best obtained output.
    """
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    data_imputed = imputer.fit_transform(data_missing_nans)
    net = MIDA_model(input_size=data_imputed.shape[1], num_layers=num_layers, theta=theta).train().type(dtype)
    optimizer = torch.optim.SGD(net.parameters(), lr=.01, momentum=.99, nesterov=True)
    loss_function = torch.nn.MSELoss().type(dtype)
    loss_hist = []
    best_net_dict = None
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        total_loss = 0
        net_input_splitted = split_batches(data_imputed.copy(), nb_batches)
        for net_input in net_input_splitted:
            input_var = torch.from_numpy(net_input).type(dtype)
            out = net(input_var)
            loss = loss_function(out, input_var)
            loss.backward()
            total_loss += loss.item()
        total_loss /= nb_batches
        if best_net_dict is None or total_loss < min(loss_hist):
            best_net_dict = net.state_dict()
        loss_hist.append(total_loss)
        if total_loss <= 1e-6:
            break
        optimizer.step()
    print(f'Stop training at epoch: {epoch + 1}/{num_epochs}, return best output')
    net.load_state_dict(best_net_dict)
    net.eval()
    best_output = net(torch.from_numpy(data_imputed).type(dtype)).detach().cpu().numpy().squeeze()
    return best_output

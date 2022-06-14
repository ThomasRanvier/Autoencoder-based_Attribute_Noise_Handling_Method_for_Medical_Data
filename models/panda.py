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

import numpy as np

from sklearn.preprocessing import KBinsDiscretizer


def panda(X: object, n_bins: int = 10, func: str = 'sum'):
    """
    Implementation of the Panda algorithm such as described by Van Hulse et al.

    Args:
        X (torch Tensor): The data.
        n_bins (int): The number of bins. Defaults to 10.
        func (str): The aggregation function. Defaults to 'sum'. Choose from {'sum', 'max'}.

    Returns:
        A numpy array containing the noise score of each instance.
    """
    binning = KBinsDiscretizer(n_bins, encode='ordinal', strategy='uniform')
    X_bin = binning.fit_transform(X).astype(int)
    S = np.zeros((X.shape[0], X.shape[1], X.shape[1]))
    for j in range(X.shape[1]):
        bin_idcs = [np.where(X_bin[:, j] == bin_i)[0] for bin_i in range(n_bins)]
        for k in range(X.shape[1]):
            if j != k:
                diff = [np.abs(X[:, k] - X[bin_idcs[bin_i], j].mean()) for bin_i in range(n_bins)]
                for i in range(X.shape[0]):
                    S[i, k, j] = abs(X[i, k] - diff[X_bin[i, j]].mean()) / diff[X_bin[i, j]].std()
    final_S = []
    if func == 'sum':
        for i in range(S.shape[0]):
            final_S.append(S[i].sum())
    if func == 'max':
        for i in range(S.shape[0]):
            final_S.append(S[i].max())
    return np.array(final_S)

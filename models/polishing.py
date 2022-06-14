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

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor

from models.panda import panda
from utils import *


def polishing(X: object, y: object, mode: str = 'standard', threshold: float = .1,
              classifier: str = 'knn', k: int = 5, filter_rate: float = .15, n_bins: int = 10, func: str = 'sum',
              random_state: int = 42):
    """
    Implementation of the polishing method such as described by Teng, that aims to correct noisy elements.
    This method has 2 modes:
        - Standard polishing: correct all misclassified elements.
        - Panda polishing: correct the filter_rate percent most noisy instances.
    Args:
        X (torch Tensor): The data to denoise.
        y (torch Tensor): The data labels.
        mode (str): Polishing mode. Defaults to 'standard'. Choose from {'standard', 'panda'}
        threshold (float): The threshold corresponding to the minimal difference necessary between the prediction and
                           the original value to replace it. Defaults to 0.1.
        classifier (str): The classifier to use for the evaluation of the method. Defaults to 'knn'
        k (int): The number of nearest neighbors if the classifier is 'knn'. Defaults to 5.
        filter_rate (float): The filtering rate when using Panda mode. Defaults to 0.15.
        n_bins (int): The number of bins when using Panda mode. Defaults to 10.
        func (str): The aggregation function when using Panda mode. Defaults to 'sum'. Choose from {'sum', 'max'}.
        random_state: (int): A random seed. Defaults to 42.

    Returns:
        A numpy array containing the obtained Accuracies and a float: the mean of the obtained RMSEs.
    """
    cv = StratifiedKFold(n_splits=k)
    if mode == 'standard':
        X_shuffled = shuffle(X, random_state)
        y_shuffled = shuffle(y, random_state)
    elif mode == 'panda':
        noise_ranking = panda(X, n_bins=n_bins, func=func)
        noise_ranking_idcs = noise_ranking.argsort()[::-1]
        sorted_X = X[noise_ranking_idcs]
        sorted_y = y[noise_ranking_idcs]
        sorted_idcs = np.array([i for i in range(X.shape[0])])
        X_shuffled = shuffle(sorted_X, random_state)
        y_shuffled = shuffle(sorted_y, random_state)
        idcs_shuffled = shuffle(sorted_idcs, random_state)
    acc_hist, auc_hist = [], []
    for train_indices, test_indices in cv.split(X_shuffled, y_shuffled):
        X_train = X_shuffled[train_indices]
        y_train = y_shuffled[train_indices]
        X_test = X_shuffled[test_indices]
        y_test = y_shuffled[test_indices]

        # Step 1: Extract elements to polish
        if mode == 'standard':
            sub_cv = StratifiedKFold(n_splits=k)
            misclassified_elements = []
            # Perform k-fold
            for sub_train_indices, sub_test_indices in sub_cv.split(X_train, y_train):
                X_sub_train, X_sub_test = X_train[sub_train_indices], X_train[sub_test_indices]
                y_sub_train, y_sub_test = y_train[sub_train_indices], y_train[sub_test_indices]
                cla = get_classifier(classifier, k=k, random_state=random_state)
                cla.fit(X_sub_train, y_sub_train)
                y_pred = cla.predict(X_sub_test)
                misclassified_elements.extend(
                    [list(a) for a in X_train[sub_test_indices[np.where(y_pred != y_sub_test)[0]]]])
            # Extract misclassified elements indices from original X_train
            to_polish = [i for i in range(X_train.shape[0]) if list(X_train[i]) in misclassified_elements]
        elif mode == 'panda':
            # Remove most noisy instances
            to_polish = []
            for i, idc in enumerate(idcs_shuffled[train_indices]):
                if idc <= filter_rate * X.shape[0]:
                    to_polish.append(i)
        # Extract elements to train on and those to predict
        sub_test_indices = to_polish
        sub_train_indices = [i for i in range(X_train.shape[0]) if i not in to_polish]

        # Step 2: Start polishing process
        polished_X_train = X_train.copy()
        replaced = 0
        if sub_test_indices:
            for i in range(X_train.shape[1]):
                # For each feature in X: swap Fi with y, split to be polished from others, regression, replace in test
                temp_X = X_train.copy()
                # Add y to X
                temp_X = np.c_[temp_X, y_train]
                # Swap y with Fi
                temp_X[:, [i, -1]] = temp_X[:, [-1, i]]
                temp_y = temp_X[:, -1]
                # Delete feature to predict from temp_X
                temp_X = np.delete(temp_X, -1, 1)
                # Regression
                reg = DecisionTreeRegressor(max_depth=5, random_state=random_state)
                reg.fit(temp_X[sub_train_indices], temp_y[sub_train_indices])
                y_pred = reg.predict(temp_X[sub_test_indices])
                y_ori = polished_X_train[sub_test_indices, i]
                # Create replacement mask depending on threshold
                y_replacement = np.where(np.abs(y_ori - y_pred) >= threshold, y_pred, y_ori)
                replaced += np.sum(np.where(np.abs(y_ori - y_pred) >= threshold, 1, 0))
                # Replace predicted feature in polished_X
                polished_X_train[sub_test_indices, i] = y_replacement
        # Step 3: Train on polished training set and evaluate on test set with same classifier as for all other methods
        cla = get_classifier('tree', random_state=random_state)
        cla.fit(polished_X_train, y_train)
        acc = balanced_accuracy_score(y_test, cla.predict(X_test))
        y_t = y_test.astype(int)
        y_oh = np.eye(np.max(y_t) + 1)[y_t][:,np.min(y_t):]
        auc = roc_auc_score(y_oh, cla.predict_proba(X_test), multi_class='ovo')
        acc_hist.append(acc)
        auc_hist.append(auc)
    return np.array(acc_hist), np.array(auc_hist)

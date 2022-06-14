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
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE


def load_covid():
    data = pd.read_excel('data/covid/covid_original.xlsx', index_col=[0, 1])
    data = data.dropna(thresh=6)
    data = data.reset_index(level=1)
    n_days=1
    dropna=False
    subset=None
    time_form='diff'
    t_diff = data['出院时间'].dt.normalize() - data['RE_DATE'].dt.normalize()
    data['t_diff'] = t_diff.dt.days.values // n_days * n_days
    data = data.set_index('t_diff', append=True)
    data = (
        data
        .groupby(['PATIENT_ID', 't_diff']).ffill()
        .groupby(['PATIENT_ID', 't_diff']).last()
    ).groupby('PATIENT_ID').tail(1)
    if dropna:
        data = data.dropna(subset=subset)
    if time_form == 'timestamp':
        data = (
            data
            .reset_index(level=1, drop=True)
            .set_index('RE_DATE', append=True)
        )
    elif time_form == 'diff':
        data = data.drop(columns=['RE_DATE'])
    ## Outcome: '出院方式'
    y = data['出院方式'].values.astype(int)
    ## Drop outcome and dates from data
    data = data.drop(columns=['入院时间', '出院时间', '出院方式'])
    data = data.apply(pd.to_numeric, errors='coerce')
    missing_mask = np.where(data.values != data.values, 0, 1)
    data_missing = data.fillna(value=data.mean()).values
    data_missing[-10, -11] = 0. ## Don't why it does not work here but no time to fix
    data_missing = ((data_missing - data_missing.min(0)) / data_missing.ptp(0)).astype(np.float32)
    data_missing = data_missing * missing_mask
    # 76 features, pad to 80 for convs
    data_missing = np.concatenate((data_missing, data_missing[:, :4]), axis=1)
    missing_mask = np.concatenate((missing_mask, missing_mask[:, :4]), axis=1)
    print(f'Dataset shape: {data_missing.shape}')
    print(f'{round(100. * np.sum(~missing_mask.astype(bool)) / np.prod(data_missing.shape), 2)}% missing data')
    print(f'Class distribution: {np.unique(y, return_counts=True)}')
    return data_missing, missing_mask, y


def load_miocardial():
    """
    0 alive, 1 dead
    """
    df = pd.read_csv('data/miocardial_infarction/miocardial_infarction.data', sep=',', header=None)
    df = df.apply(pd.to_numeric, errors='coerce')
    y = df.iloc[:, 112:]
    y = np.where(y.to_numpy()[:, -1].astype(int) >= 1, 1, 0)
    df = df.drop([0, 7, 34, 35, 88, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123], axis=1)
    missing_mask = np.where(df.values != df.values, 0, 1)
    data_missing = df.fillna(value=df.mean()).values
    data_missing = ((data_missing - data_missing.min(0)) / data_missing.ptp(0)).astype(np.float32)
    data_missing = data_missing * missing_mask
    # 107 features, pad to 110 for convs
    data_missing = np.concatenate((data_missing[:, :3], data_missing), axis=1)
    missing_mask = np.concatenate((missing_mask[:, :3], missing_mask), axis=1)
    print(f'Dataset shape: {data_missing.shape}')
    print(f'{round(100. * np.sum(~missing_mask.astype(bool)) / np.prod(data_missing.shape), 2)}% missing data')
    print(f'Class distribution: {np.unique(y, return_counts=True)}')
    return data_missing, missing_mask, y


def load_nhanes(balanced=True):
    ## Load original dataset splitted by views
    data_raw, y, views, num_elements = load_mat_dataset('NHANES')
    ## Concat views
    data = np.concatenate(data_raw, -1)
    ## Manually select 1000 random samples from class 1 and 1000 random samples from class 2
    data = shuffle(data, 0)
    y = shuffle(y, 0)
    class_1_indices = np.random.choice(np.where(y == 1)[0], size=1000 if balanced else 100, replace=False)
    class_2_indices = np.random.choice(np.where(y == 2)[0], size=1000 if balanced else 1900, replace=False)
    data = np.concatenate([data[class_1_indices, :], data[class_2_indices, :]])
    y = np.concatenate([y[class_1_indices], y[class_2_indices]])
    data = shuffle(data, 1)
    y = shuffle(y, 1)
    ## Drop useless columns
    data = pd.DataFrame(data)
    for col in data.columns:
        if len(data[col].unique()) == 1:
            data.drop(col, inplace=True, axis=1)
    missing_mask = np.where(data.values != data.values, 0, 1)
    n = np.sum(~missing_mask.astype(bool))
    ## Normalize
    data = data.fillna(value=data.mean()).values
    data = ((data - data.min(0)) / data.ptp(0)).astype(np.float32)
    ## Replace missing values by 0
    data = data * missing_mask
    data_missing = data
    print(f'Dataset shape: {data_missing.shape}')
    print(f'{round(100. * n / np.prod(data.shape), 2)}% missing data')
    print(f'Class distribution: {np.unique(y, return_counts=True)}')
    return data_missing, missing_mask, y


def compute_rmse(corrected: object, original: object, n: int):
    """
    Compute the RMSE metric between the obtained correction and the true data.

    Args:
        corrected (numpy array): Corrected data.
        original (numpy array): True data.
        n (int): Number of corrupted values.

    Returns:
        A float, the RMSE score.
    """
    return (np.linalg.norm(corrected - original, 'fro') ** 2 / n) ** 0.5


def shuffle(array: object, seed: int):
    """
    A method to shuffle a given list on the 1st dimension given a random seed.

    Args:
        array (numpy array): The array to shuffle.
        seed (int): The random seed to use.

    Returns:
        The shuffled numpy array.
    """
    shuf_order = np.arange(len(array))
    np.random.seed(seed)
    np.random.shuffle(shuf_order)
    array_shuffled = array.copy()[shuf_order]
    return array_shuffled


def unshuffle(array: object, seed: int):
    """
    A method to unshuffle a given list on the 1st dimension that have been previously shuffled given the seed.

    Args:
        array (numpy array): The array to unshuffle.
        seed (int): The random seed to use.

    Returns:
        The unshuffled numpy array.
    """
    shuf_order = np.arange(len(array))
    np.random.seed(seed)
    np.random.shuffle(shuf_order)
    unshuf_order = np.zeros_like(shuf_order)
    unshuf_order[shuf_order] = np.arange(len(array))
    array_unshuffled = array.copy()[unshuf_order]
    return array_unshuffled


def inject_noise(data: object, noise_rate: float):
    """
    A method to inject noise at a given rate in the data.

    Args:
        data (numpy array): The data to add noise.
        noise_rate (float): The noise rate.

    Returns:
        A numpy array, the noisy data.
    """
    data_noisy = np.random.random(size=data.shape)
    noise_mask = np.random.random(size=data.shape)
    noise_mask = np.where(noise_mask < noise_rate, 1, 0).astype(bool)
    data_noisy *= noise_mask
    data_noisy += data.copy() * ~noise_mask
    return data_noisy


def get_classifier(classifier: str, k: int = 5, max_depth: int = None, min_samples_leaf: int = 10,
                   ccp_alpha: float = 0., random_state: int = 42):
    """
    Gives the classifier corresponding to the given parameters.

    Args:
        classifier (str): Choose from {'knn', 'adaboost', 'tree'}.
        k (int): The number of nearest neighbors if the classifier is 'knn'. Defaults to 5.
        max_depth (int): The max tree depth if the classifier is 'tree'. Defaults to None.
        min_samples_leaf (int): Used if the classifier is 'tree'. Defaults to 10.
        ccp_alpha (float): Used if the classifier is 'tree'. Defaults to 0.
        random_state (int): The random seed used to initialize the classifier. Defaults to 42.

    Returns:
        The scikit-learn classifier.
    """
    if classifier == 'knn':
        cla = KNeighborsClassifier(k)
    elif classifier == 'adaboost':
        cla = AdaBoostClassifier(random_state=random_state)
    elif classifier == 'tree':
        cla = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                     ccp_alpha=ccp_alpha, random_state=random_state)
    return cla


def get_scores(X: object, y: object, classifier: str = 'tree', shuffle_state: int = 42, k: int = 5,
               max_depth: int = None, min_samples_leaf: int = 10, ccp_alpha: float = 0.):
    X = shuffle(X, shuffle_state)
    y = shuffle(y, shuffle_state)
    cla = get_classifier(classifier, k=k, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                               ccp_alpha=ccp_alpha, random_state=shuffle_state)
    scores = cross_validate(cla, X, y, cv=5, scoring=('balanced_accuracy', 'roc_auc_ovo'), error_score='raise')
    return scores


def split_batches(array: object, nb_batches: int):
    """
    Split the given array in nb_batches batches.

    Args:
        array: The array to split on the 1st dimension.
        nb_batches: Number of batches to create.

    Returns:
        The data split in batches.
    """
    result = []
    split = int(array.shape[0] / nb_batches)
    for i in range(nb_batches):
        if i == nb_batches - 1:
            result.append(array)
        else:
            result.append(array[:split])
            array = array[split:]
    return result


def get_details(dataset):
    """
    Give the views names and dimensions for multi-view datasets.

    Args:
        dataset (str): The dataset name.

    Returns:
        A dictionary of the form {'view name': view dimension} and the number of elements in the dataset.
    """
    if 'NHANES' in dataset:
        views = [('df', 56), ('lab', 21), ('exam', 6), ('quest', 14)]
        num_elements = 24369
    return views, num_elements


def load_mat_dataset(dataset):
    """
    Load a .mat dataset given its name.

    Args:
        dataset (str): The dataset name.

    Returns:
        The data and the labels, both in numpy arrays, and a dictionary of the form {'view name': view dimension} and
        the number of elements in the dataset.
    """
    views, num_elements = get_details(dataset)
    mat_contents = scipy.io.loadmat(f'data/{dataset}.mat')
    y = mat_contents['y'].squeeze()
    X = [None] * len(views)
    for i in range(len(views)):
        dv = mat_contents['X'][0, i].shape[1]
        view_index = [view[1] for view in views].index(dv)
        X[view_index] = mat_contents['X'][0, i].astype(np.float32)
    return X, y, views, num_elements

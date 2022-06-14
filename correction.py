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

import time
import random
import torch
import numpy as np
import os

import utils
from models.dense import Dense
from models.skip import Skip


def run(data_corrupted, num_iter, params, missing_mask=None,
        y=None, data_true=None, min_ite=0, eval_every=1,
        cuda=True, training_idcs=None, show_every=50, seed=42):
    """
    Implementation of our autoencoder-based method.

    Args:
        data_corrupted (numpy array): The original corrupted data.
        num_iter (int): The maximum number of iterations that the method will perform.
        params (dict): The run parameters:
           {
               'nb_batches': how many batches to split the data for,
               'reg_noise_std': amount of noise to add to the net_input at each iteration,
               'net_input': either 'data_corrupted' or 'noise', determines if the net is given pure noise or the corrupted data as input,
               'net_params': the model parameters, either a list containing the layers sizes to use a dense model, or a dictionary containing params for the Skip model, cf. models.skip,
               'adam_lr': learning rate value,
               'adam_weight_decay': weight decay value,
           }
        missing_mask (numpy array): A boolean matrix indicating missing values in the corrupted data, 1 if the value is known, 0 if it is unknown.
        y (numpy array): The data labels.
        data_true (numpy array): The true data, if it is given the RMSE between the true data and the corrected one will be computed and displayed.
        min_ite (int): Number of iterations that will be executed before evaluation starts.
        eval_every (int): Interval of iterations after which to perform evaluation.
        cuda (bool): Boolean to determine if we use Cuda or not.
        training_idcs (numpy array): Indices of the elements that will be used for evaluation.
        show_every (int): Interval of iterations after which infos are logged.
        seed (int): The random seed to use.
        
    Returns:
        A dictionnary:
        {
            'raw_out': The raw correction.
            'masked_out': When a missing values mask is given, the correction masked with known values to only keep the model imputations.
            'i': The iteration at which the best correction has been saved.
            'ACC': The balanced accuracy score of the best correction.
            'AUC': The AUC score of the best correction.
            'loss': The loss on the best correction.
        }
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    float_dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    bool_dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    ## Params
    nb_batches = params['nb_batches']
    reg_noise_std = params['reg_noise_std']
    net_input = params['net_input']
    net_params = params['net_params']
    adam_lr = params['adam_lr']
    adam_weight_decay = params['adam_weight_decay']

    ## Net input
    net_input = np.random.random(size=data_corrupted.shape) if net_input == 'noise' else data_corrupted

    ## Model
    if type(net_params) == dict:
        net = Skip(**net_params).type(float_dtype)
    else:
        net = Dense(input_size=data_corrupted.shape[1],
                    neurons=net_params).type(float_dtype)

    ## Optimizer
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=adam_lr,
                                 weight_decay=adam_weight_decay)

    ## Loss function
    loss_function = torch.nn.MSELoss().type(float_dtype)

    ## Training process
    # Convert variables to the right numpy dtype, Float32 for all and Byte for the mask
    target_np = data_corrupted.astype(np.float32)
    if missing_mask is not None:
        mask_np = missing_mask.astype('?')
    net_input_np = net_input.astype(np.float32)

    # Create global variables
    best_correction = {}
    no_improve = 0
    acc_hist, auc_hist, loss_hist = [], [], []

    for i in range(num_iter):
        optimizer.zero_grad()
        start = time.time()
        # Add the noise regularization to net_input
        net_input_shuffled = utils.shuffle(net_input_np, i)
        if reg_noise_std > 0:
            net_input_shuffled += np.random.normal(size=net_input_shuffled.shape) * reg_noise_std
        # target_np and mask_np shuffle
        target_shuffled = utils.shuffle(target_np, i)
        if missing_mask is not None:
            mask_shuffled = utils.shuffle(mask_np, i)
        # Add channel axis: (batch, channel, x)
        net_input_shuffled = net_input_shuffled[:, None, :]
        target_shuffled = target_shuffled[:, None, :]
        if missing_mask is not None:
            mask_shuffled = mask_shuffled[:, None, :]
        # Split: net_input, target_np, mask_np
        net_input_splitted = utils.split_batches(net_input_shuffled.copy(), nb_batches)
        target_splitted = utils.split_batches(target_shuffled.copy(), nb_batches)
        if missing_mask is not None:
            mask_splitted = utils.split_batches(mask_shuffled.copy(), nb_batches)
        ## Train one batch
        loss = 0
        concat_out = []
        for batch in range(nb_batches):
            # Put variables of the batch on CUDA
            net_input = torch.from_numpy(net_input_splitted[batch]).type(float_dtype)
            target_var = torch.from_numpy(target_splitted[batch]).type(float_dtype)
            if missing_mask is not None:
                mask_var = torch.from_numpy(mask_splitted[batch]).type(bool_dtype)
            # Get net output
            out = net(net_input)
            # Compute loss with different loss term depending on correction mode
            if missing_mask is not None:
                total_loss = loss_function(out * mask_var, target_var)
            else:
                total_loss = loss_function(out, target_var)
            # Perform back prop
            total_loss.backward()
            loss += total_loss.item()
            concat_out.append(out.detach().cpu().numpy().squeeze())
        loss /= nb_batches
        loss_hist.append(loss)
        # Concatenate outputs split by batches and unshuffle given the seed used for shuffling
        unshuffled_out = utils.unshuffle(np.concatenate(concat_out, axis=0), i)
        if missing_mask is not None:
            out_masked_by_truth = np.where(mask_np, target_np, unshuffled_out)
        # Compute RMSE
        if data_true is not None:
            rmse = utils.compute_rmse(unshuffled_out, data, np.prod(data_corrupted.shape))
            # If defined, mask the output with known values
            if missing_mask is not None:
                masked_rmse = utils.compute_rmse(out_masked_by_truth, data, np.sum(~missing_mask.astype(np.bool)))
        ## Evaluation, check of stopping conditions, saving results if best ever
        ## obtained, etc.
        if y is not None:
            if i >= min_ite and i % eval_every == 0:
                scores = utils.get_scores(unshuffled_out[training_idcs].squeeze(),
                                          y[training_idcs].squeeze())
                acc = scores['test_balanced_accuracy'].mean()
                auc = scores['test_roc_auc_ovo'].mean()
                if missing_mask is not None:
                    scores_masked = utils.get_scores(out_masked_by_truth[training_idcs].squeeze(),
                                                  y[training_idcs].squeeze())
                    acc = max(acc, scores_masked['test_balanced_accuracy'].mean())
                    auc = max(auc, scores_masked['test_roc_auc_ovo'].mean())
                acc = round(acc * 100, 2)
                auc = round(auc * 100, 2)
                acc_hist.append(acc)
                auc_hist.append(auc)
                if len(auc_hist) >= 2:
                    no_improve = no_improve + 1 if max(auc_hist[:-1]) >= auc_hist[-1] else 0
                    if max(auc_hist[:-1]) < auc_hist[-1]:
                        best_correction['raw_out'] = unshuffled_out
                        if missing_mask is not None:
                            best_correction['masked_out'] = out_masked_by_truth
                        best_correction['i'] = i
                        best_correction['acc'] = acc
                        best_correction['auc'] = auc
                else:
                    best_correction['raw_out'] = unshuffled_out
                    if missing_mask is not None:
                        best_correction['masked_out'] = out_masked_by_truth
                    best_correction['i'] = i
                    best_correction['acc'] = acc
                    best_correction['auc'] = auc
        if y is None:
            if len(loss_hist) >= 2:
                no_improve = no_improve + 1 if min(loss_hist[:-1]) < loss_hist[-1] else 0
                if min(loss_hist[:-1]) > loss_hist[-1]:
                    best_correction['raw_out'] = unshuffled_out
                    if missing_mask is not None:
                        best_correction['masked_out'] = out_masked_by_truth
                    best_correction['i'] = i
                    best_correction['loss'] = loss
            else:
                best_correction['raw_out'] = unshuffled_out
                if missing_mask is not None:
                    best_correction['masked_out'] = out_masked_by_truth
                best_correction['i'] = i
                best_correction['loss'] = loss
        ## Check for early stop
        early_stop = no_improve >= max(10, num_iter // 10)
        ## Prepare print
        elapsed_time = round(time.time() - start, 2)
        info = f'Ite {i:05d} - {elapsed_time:.2f} sec - Loss {loss:.6f} - '
        if data_true is not None:
            info += f'RMSE {rmse:.6f} - '
            if missing_mask is not None:
                info += f'Masked RMSE {masked_rmse:.6f} - '
        if y is not None:
            info += f'ACC {acc:03.2f}% - ACC Mean {np.mean(acc_hist[-show_every:]):03.2f}% - '
            info += f'AUC {auc:03.2f}% - AUC Mean {np.mean(auc_hist[-show_every:]):03.2f}% - '
        info += f'Deter {no_improve:03d}'
        if i % show_every == 0 or early_stop:
            print(info)
        else:
            print(info, '\r', end='')
        if early_stop:
            if y is None:
                s = f'loss of {best_correction["loss"]}'
            else:
                s = f'acc of {best_correction["acc"]}% and auc of {best_correction["auc"]}%'
            print(f'Early stop ite {i}, rollback to correction of ite {best_correction["i"]}' +
                  f', whith {s}')
            break
        optimizer.step()
    return best_correction
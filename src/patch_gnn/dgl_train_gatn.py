#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Modules for training and assessing the performance of different models"""

from datetime import datetime
from typing import Tuple

import dgl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from patch_gnn.dgl_cv_split import k_fold_split
from patch_gnn.dgl_dataset import PatchGNNDataset, collate_fn
from patch_gnn.dgl_layer import GATN


def reset_weights(m):
    """
    Reset model weights to avoid
    weight leakage.

    m: a pytorch model
    """
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()


def cv_on_train_with_gatn(
    train_dataset: PatchGNNDataset,
    lr: float = 1e-4,
    k_fold: int = 3,
    n_epochs: int = 100,
    in_dim: int = 67,
    embed_dim: int = 96,
    gat_n_heads: int = 1,
    gat_out_dim: int = 64,
    dense_out1: int = 128,
    dense_out2: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Split trian data into k folds, (k-1) folds used for training,
    1 fold used for validation. Each fold is trained n_epochs and
    and validated after each epoch. Weightes are reset for each cv fold.
    Loss of each cross validation fold is averaged and plotted.

    :param train_dataset: a PatchGNNDataset
    :param lr: learning rate
    :param k_fold: the number of cross validation fold
    :param n_epochs: the number of epochs
    :param in_dim: the input dimension, usually is 67 for 67 descriptors
    :param embed_dim: the embedding dim
    :param gat_n_heads: number of heads for attention models
    ...

    returns:
    disaplys a matlibplot with average training and validation error for each epoch
    a np.ndarray indicating the average cv in training fold per epoch
    a np.ndarray indicating the average cv in validation fold per epoch

    """
    # init model
    net = GATN(
        in_dim=in_dim,
        embed_dim=embed_dim,
        gat_n_heads=gat_n_heads,
        gat_out_dim=gat_out_dim,
        dense_out1=dense_out1,
        dense_out2=dense_out2,
        dense_out3=1,
    )

    # create optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # get loss function
    mse_loss = nn.MSELoss()

    # do cross validation here
    all_train_folds = k_fold_split(train_dataset, k=k_fold)

    cv_train_mseloss = {}
    cv_val_mseloss = {}

    for fold, (train_fold, val_fold) in enumerate(all_train_folds):

        # reset weights for each fold
        net.apply(reset_weights)

        # print(f"training {fold} fold")
        #  loader to load all train set
        train_loader = GraphDataLoader(
            train_fold, batch_size=len(train_fold), collate_fn=collate_fn
        )
        # loader to load all validation set
        val_loader = GraphDataLoader(
            val_fold, batch_size=len(val_fold), collate_fn=collate_fn
        )

        # start epochs here, for each train/val split, do epoch
        for epoch in tqdm(range(n_epochs)):
            # init training loss
            train_mse_loss = 0.0
            # start training
            # one fold training
            for train_graphs, train_targets in train_loader:
                pred_train_target, _ = net(
                    train_graphs, train_graphs.ndata["feat"]
                )

                # both pred and train labels are tensor.size([1])
                actual_loss = mse_loss(
                    pred_train_target.float(),
                    train_targets.float().unsqueeze(1),
                )

                # backward
                # need to set gradient to zero first https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
                optimizer.zero_grad()
                actual_loss.backward()
                optimizer.step()

                # update train loss
                train_mse_loss += actual_loss.detach().numpy()

            # update cv dict
            if fold not in cv_train_mseloss:
                cv_train_mseloss[fold] = list()

            cv_train_mseloss[fold].append(train_mse_loss)

            # after each epoch, validate
            # since we're not training, we don't need to calculate the gradients for our outputs
            # print(f"evaluate on {fold} fold")
            with torch.no_grad():
                val_mse_loss = 0.0
                for val_graphs, val_targets in val_loader:
                    pred_val_target, _ = net(
                        val_graphs, val_graphs.ndata["feat"]
                    )
                    # both pred and train labels are tensor.size([1])
                    val_actual_loss = mse_loss(
                        pred_val_target.float(),
                        val_targets.float().unsqueeze(1),
                    )

                # update val loss
                val_mse_loss += val_actual_loss.detach().numpy()

                # update cv dict
                if fold not in cv_val_mseloss:
                    cv_val_mseloss[fold] = list()

                cv_val_mseloss[fold].append(val_mse_loss)

    avg_loss_per_epoch_train = (
        pd.DataFrame(cv_train_mseloss).mean(axis=1).to_numpy()
    )
    avg_loss_per_epoch_val = (
        pd.DataFrame(cv_val_mseloss).mean(axis=1).to_numpy()
    )

    # make performance plot
    plt.plot(
        list(range(n_epochs)),
        avg_loss_per_epoch_train,
        "g",
        label="Average CV training loss",
    )
    plt.plot(
        list(range(n_epochs)),
        avg_loss_per_epoch_val,
        "b",
        label="Average CV validation loss",
    )
    plt.title("CV Training and Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.show()

    return avg_loss_per_epoch_train, avg_loss_per_epoch_val


def train_val_with_gatn_no_cv(
    lr_lst: list,
    train_loader: GraphDataLoader,
    test_loader: GraphDataLoader,
    save_checkpoint: bool = False,
    tensorboard: bool = False,
    early_stopping:bool = False,
    n_epochs: int = 100,
    in_dim: int = 67,
    embed_dim: int = 96,
    gat_n_heads: int = 1,
    gat_out_dim: int = 64,
    dense_out1: int = 128,
    dense_out2: int = 64,
):
    """
    Perform dgl gatn training and evaluation without cross
    validation. With the option to save the training process
    values in and visualize it in tensorboard

    Args:
        lr_lst (list): a list with learning rate
        train_loader (GraphDataLoader): a data loader to load train set
        test_loader (GraphDataLoader): a data loader to load test set
        save_checkpoint(bool, optional): whether to save the model after all epochs into file. Defaults to False.
        tensorboard (bool, optional): whether to save the performance for display in tensorboard.
                                     Defaults to False.
        early_stopping (bool, optional): whether to use early stopping to stop training the model up to X epoch based 
                                        on the performance diff between training set and test set. 
                                        Defaults to False.
        n_epochs (int, optional): number of epochs. Defaults to 100.
        in_dim (int, optional): the input dimension, usually is 67 for 67 descriptors.
                                 Defaults to 67.
        embed_dim (int, optional): embedding dim. Defaults to 96.
        gat_n_heads (int, optional): number of attention heads. Defaults to 1.
        gat_out_dim (int, optional): dimension out from attention. Defaults to 64.
        dense_out1 (int, optional): 1st FC layer dimension. Defaults to 128.
        dense_out2 (int, optional): 2nd FC layer dimension. Defaults to 64.
    """

    for lr in lr_lst:
        now = datetime.now()
        current_time = now.strftime("%m-%d-%Y-%Hh%M")

        if tensorboard:
            writer = SummaryWriter(
                f"runs/lr_{lr}_{n_epochs}epoch_{current_time}"
            )
        # create the model
        net = GATN(
            in_dim=in_dim,
            embed_dim=embed_dim,
            gat_n_heads=gat_n_heads,
            gat_out_dim=gat_out_dim,
            dense_out1=dense_out1,
            dense_out2=dense_out2,
            dense_out3=1,
        )

        # create optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        # training main loop
        train_mseloss_list = list()
        test_mseloss_list = list()

        for epoch in tqdm(range(n_epochs)):
            # start training
            # batch size is 251 here
            for train_graphs, train_targets in train_loader:
                pred_train_target, atten = net(
                    train_graphs, train_graphs.ndata["feat"]
                )

                # compute loss for each batch - shouldn't compute loss for the entire dataset, right?
                mse_loss = nn.MSELoss()
                # both pred and train labels are tensor.size([1])
                actual_loss = mse_loss(
                    pred_train_target.float(),
                    train_targets.float().unsqueeze(1),
                )

                # backward
                # need to set gradient to zero first https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
                optimizer.zero_grad()
                actual_loss.backward()
                optimizer.step()

                # pred_lst.append(pred_train_target.detach().numpy())
                train_mseloss_list.append(actual_loss.detach().numpy())

            if tensorboard:
                writer.add_scalar(
                    "mean_MSELoss/train", np.mean(train_mseloss_list), epoch
                )
                writer.add_text("lr", f"learning rate {lr}")
                writer.add_scalar(
                    "variance explained/train",
                    evs(train_targets, pred_train_target.detach().numpy()),
                    epoch,
                )

            # predict
            for test_graphs, test_targets in test_loader:

                pred_test_target, atten_test = net(
                    test_graphs, test_graphs.ndata["feat"]
                )
                # compute loss for each batch - shouldn't compute loss for the entire dataset, right?
                mse_loss = nn.MSELoss()

                # both pred and train labels are tensor.size([1])
                actual_test_loss = mse_loss(
                    pred_test_target.float(), test_targets.float().unsqueeze(1)
                )
                test_mseloss_list.append(actual_test_loss.detach().numpy())

                # print(f"mseloss_test_list is {mseloss_test_list}")
            # print(f"len of pred is {len(pred_test_lst)}")
            if tensorboard:
                writer.add_scalar(
                    "mean_MSELoss/test", np.mean(test_mseloss_list), epoch
                )
                writer.add_text("lr", f"learning rate {lr}")
                writer.add_scalar(
                    "variance explained/test",
                    evs(test_targets, pred_test_target.detach().numpy()),
                    epoch,
                )

            # early stopping if performance on training set >> performance on testset
            if early_stopping:
                # if training is much better than test performance, stop training more epochs
                if train_mseloss_list[-1] - test_mseloss_list[-1] >0.4:
                    break

        if save_checkpoint:
            print(f'epoch {epoch}, learning rate {lr}, training \
            mseloss {train_mseloss_list[-1]}, \
            test mseloss {test_mseloss_list[-1]}')
            torch.save(net, f"best_gatn_{lr}.pkl")

    if tensorboard:
        writer.flush()
        writer.close()


def get_attention_plot(
    test_dataset: PatchGNNDataset, idx: int, attention_list: list
) -> np.ndarray:
    """
    for a given graph, draw attention plot and networkx plot on test set

    :param test_dataset: a PatchGNNDataset object, of length n,
                        each index is a tuple[dgl.graph, a dictionary of nodes,
                        graph target value]
    :param idx: the index from PatchGNNDataset one wants to plot the attention on
    :param attention_list: a list containing attention values of each graph,
                            should be the same length as test_dataset

    return: a, the attention matrix

    """
    # sort the node order from test_dataset
    labels = list(sorted(test_dataset[idx][1].values()))
    # https://stackoverflow.com/questions/2318529/plotting-only-upper-lower-triangle-of-a-heatmap
    # make a nxn attention grid
    a = np.zeros(
        (len(labels), len(labels))
    )  # default_rng(42).random((len(labels),len(labels)))
    # fill in attention grid with edge attention info
    for i in range(len(test_dataset[idx][0].edges(form="uv")[0])):
        a[
            test_dataset[idx][0].edges(form="uv")[0][i],
            test_dataset[idx][0].edges(form="uv")[1][i],
        ] = attention_list[idx][i]
    # plot attention
    plt.figure(figsize=(6, 5))
    sns.heatmap(a, xticklabels=labels, yticklabels=labels)
    # plot networkx connectivitiy
    plt.figure(figsize=(5, 5))
    dgl_to_nx = nx.relabel_nodes(
        test_dataset[idx][0].to_networkx(), mapping=test_dataset[idx][1]
    )
    nx.draw_networkx(
        dgl_to_nx,
        with_labels=True,
        node_size=500,
        node_color=[
            "lightblue" if "MET" not in i else "lightgreen"
            for i in list(test_dataset[idx][1].values())
        ],
    )

    return a

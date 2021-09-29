#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Modules for training and assessing the performance of different models"""

from datetime import datetime

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from patch_gnn.dgl_cv_split import k_fold_split
from patch_gnn.dgl_dataset import PatchGNNDataset, collate_fn
from patch_gnn.dgl_layer import GATN


def cv_on_train_with_gatn(
    train_dataset: PatchGNNDataset,
    lr: float = 1e-4,
    tensorboard: bool = False,
    k_fold: int = 3,
    save_checkpoint: bool = False,
    n_epochs: int = 100,
    in_dim: int = 67,
    embed_dim: int = 96,
    gat_n_heads: int = 1,
    gat_out_dim: int = 64,
    dense_out1: int = 128,
    dense_out2: int = 64,
):

    """
    Split
    """
    now = datetime.now()
    current_time = now.strftime("%m-%d-%Y-%Hh%M")

    if tensorboard:
        writer = SummaryWriter(f"runs/lr_{lr}_{n_epochs}epoch_{current_time}")

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
    val_mseloss_list = list()

    mse_loss = nn.MSELoss()

    for epoch in tqdm(range(n_epochs)):
        train_mes_loss = 0.0
        val_mse_loss = 0.0
        # start training
        # cross validation
        all_train_folds = k_fold_split(train_dataset, k=k_fold)

        for train_fold, val_fold in all_train_folds:
            train_loader = GraphDataLoader(
                train_fold, batch_size=len(train_fold), collate_fn=collate_fn
            )  # , #ddp_seed =42)
            val_loader = GraphDataLoader(
                val_fold, batch_size=len(val_fold), collate_fn=collate_fn
            )  # , #ddp_seed =42)

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
                train_mes_loss += actual_loss.detach().numpy()

            # one fold validation
            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                for val_graphs, val_targets in val_loader:
                    pred_val_target, _ = net(
                        val_graphs, val_graphs.ndata["feat"]
                    )
                    # both pred and train labels are tensor.size([1])
                    val_actual_loss = mse_loss(
                        pred_val_target.float(),
                        val_targets.float().unsqueeze(1),
                    )

                    # update train loss
                    val_mse_loss += val_actual_loss.detach().numpy()

        # save checkpoint after it reach to a criteria
        if abs(val_actual_loss - actual_loss) < 1e-4:
            print(f"model reach a optimum, saving model...")
            torch.save(net, "best_gatn")

        # after all folds
        train_mseloss_list.append(train_mes_loss / k_fold)
        val_mseloss_list.append(val_mse_loss / k_fold)

        if tensorboard:
            writer.add_scalar(
                "mean_MSELoss/train", train_mseloss_list[-1], epoch
            )
            writer.add_text("lr", f"learning rate {lr}")
            # writer.add_scalar("variance explained/train", evs(train_targets, pred_train_target.detach().numpy()), epoch)
            writer.add_scalar("mean_MSELoss/val", val_mseloss_list[-1], epoch)
            writer.add_text("lr", f"learning rate {lr}")

    if tensorboard:
        writer.flush()
        writer.close()


def train_val_with_gatn_no_cv(
    lr_lst: list,
    train_loader: GraphDataLoader,
    test_loader: GraphDataLoader,
    tensorboard: bool = False,
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
        tensorboard (bool, optional): [description]. Defaults to False.
        n_epochs (int, optional): [description]. Defaults to 100.
        in_dim (int, optional): [description]. Defaults to 67.
        embed_dim (int, optional): [description]. Defaults to 96.
        gat_n_heads (int, optional): [description]. Defaults to 1.
        gat_out_dim (int, optional): [description]. Defaults to 64.
        dense_out1 (int, optional): [description]. Defaults to 128.
        dense_out2 (int, optional): [description]. Defaults to 64.
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
                test_target_lst.append(test_targets)

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

        if tensorboard:
            writer.flush()
            writer.close()

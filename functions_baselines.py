import torch
import os
import dgl
import numpy as np
import pandas as pd

def ETC_calculate_beta_loss(df, group_idx):
    """
    Calculate beta loss
    """
    # beta_loss = 0
    beta_loss_list = []
    for i, rows in df.groupby(group_idx):
        beta_loss = 2 * len(rows) - len(set(rows['src'].tolist() + rows['dst'].tolist()))
        beta_loss_list.append(beta_loss)
    
    return beta_loss_list


def ETC_single_pass_batching(df,eps):
    """
    ETC single-pass batching
    """
    Cu = 0
    batch_idx = 0
    node_set = []
    # cur_batch = []
    group_idx = []

    for i in range(len(df)):
        Cu = Cu + 2
        row = df.iloc[i]
        src_node = row['src']
        dst_node = row['dst']

        node_set.append(src_node)
        node_set.append(dst_node)
        node_set = list(set(node_set))

        beta_loss = Cu - len(node_set)

        if beta_loss < eps:
            # cur_batch.append(i)
            group_idx.append(batch_idx)
        else:
            batch_idx += 1
            Cu = 0
            # cur_batch = [i]
            node_set = list(set([src_node, dst_node]))
            group_idx.append(batch_idx)
    
    return group_idx

def ETC_efficient_data_access():
    """
    ETC efficient data access
    This is for CPU-GPU communication. In our case, all data are stored in GPU.
    """
    pass

def NS_batching(df, GF):
    """
    NS batching
    Eseentially, this is a TG-Diffuser with node count = 0, and window size = size of the total graph
    """
    pass
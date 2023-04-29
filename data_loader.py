import warnings

import os
import dgl
import torch
import numpy as np

from sklearn.model_selection import train_test_split
from utils import normalize_features

from dgl.data import FraudYelpDataset, FraudAmazonDataset
from dgl.data.utils import load_graphs, save_graphs
import pandas as pd
from pygod.utils import load_data as pygod_load_data

warnings.simplefilter("ignore")


def load_data(args):
    dataset_str = args.dataset

    if dataset_str == 'yelp':
        dataset = FraudYelpDataset()
        graph = dataset[0]

        graph = dgl.to_homogeneous(graph, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
        graph = dgl.add_self_loop(graph)

        train_mask, val_mask, test_mask = graph_split(dataset_str, graph.ndata['label'], train_ratio=args.train_ratio,
                                                      folds=args.ntrials)

        x_data = torch.tensor(normalize_features(graph.ndata['feature'], norm_row=False), dtype=torch.float)

        return x_data, graph.ndata['feature'].size()[-1], graph.ndata['label'], 2, \
            train_mask, val_mask, test_mask, graph
        # graph.ndata['train_mask'].bool(), graph.ndata['val_mask'].bool(), graph.ndata['test_mask'].bool()

    elif dataset_str == 'amazon':
        dataset = FraudAmazonDataset()
        graph = dataset[0]

        graph = dgl.to_homogeneous(graph, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
        graph = dgl.add_self_loop(graph)

        train_mask, val_mask, test_mask = graph_split(dataset_str, graph.ndata['label'], train_ratio=args.train_ratio,
                                                      folds=args.ntrials)

        graph.ndata['feature'] = torch.tensor(normalize_features(graph.ndata['feature'], norm_row=True),
                                              dtype=torch.float)

        return graph.ndata['feature'], graph.ndata['feature'].size()[-1], graph.ndata['label'], 2, \
            train_mask, val_mask, test_mask, graph

    elif dataset_str == 'reddit':
        data = pygod_load_data(dataset_str)
        graph = dgl.graph((data.edge_index[0], data.edge_index[1]))
        graph.ndata['feature'] = data.x
        graph.ndata['label'] = data.y.type(torch.LongTensor)

        train_mask, val_mask, test_mask = graph_split(dataset_str, graph.ndata['label'], train_ratio=args.train_ratio, folds=args.ntrials)

        graph.ndata['feature'] = torch.tensor(normalize_features(graph.ndata['feature'], norm_row=True), dtype=torch.float)

        return graph.ndata['feature'], graph.ndata['feature'].size()[-1], graph.ndata['label'], 2, \
            train_mask, val_mask, test_mask, graph

    else:
        raise NotImplementedError


def graph_split(dataset, labels, train_ratio=0.01, folds=5):
    """split dataset into train and test

    Args:
        dataset (str): name of dataset
        labels (list): list of labels of nodes
    """
    assert dataset in ['amazon', 'yelp', 'reddit']
    if dataset == 'amazon':
        index = np.array(range(3305, len(labels)))
        stratify_labels = labels[3305:]

    elif dataset == 'yelp' or dataset == 'reddit':
        index = np.array(range(len(labels)))
        stratify_labels = labels

    else:
        index = np.array(range(46564))
        stratify_labels = labels[:46564]

    # generate mask
    train_mask, valid_mask, test_mask = [], [], []

    for fold in range(folds):
        idx_train, idx_test = train_test_split(index,
                                               stratify=stratify_labels,
                                               train_size=train_ratio,
                                               random_state=fold,
                                               shuffle=True)

        idx_valid, idx_test = train_test_split(idx_test,
                                               stratify=np.array(labels)[idx_test],
                                               test_size=2.0 / 3,
                                               random_state=fold, shuffle=True)

        train_mask_fold = torch.BoolTensor([False for _ in range(len(labels))])
        valid_mask_fold = torch.BoolTensor([False for _ in range(len(labels))])
        test_mask_fold = torch.BoolTensor([False for _ in range(len(labels))])
        train_mask_fold[idx_train] = True
        valid_mask_fold[idx_valid] = True
        test_mask_fold[idx_test] = True

        train_mask.append(train_mask_fold)
        valid_mask.append(valid_mask_fold)
        test_mask.append(test_mask_fold)

    train_mask = torch.vstack(train_mask)
    valid_mask = torch.vstack(valid_mask)
    test_mask = torch.vstack(test_mask)

    return train_mask, valid_mask, test_mask

import dgl
import math
import time
import numpy as np
import random as rd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score
from collections import namedtuple


Evaluation_Metrics = namedtuple('Evaluation_Metrics', ['accuracy',
                                                       'macro_F1',
                                                       'recall',
                                                       'auc',
                                                       'ap',
                                                       'gmean'])

EOS = 1e-10


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    rd.seed(seed)
    torch.backends.cudnn.deterministic = True


def create_activation(name):
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "gelu":
        return nn.GELU(inplace=True)
    elif name == "prelu":
        return nn.PReLU(inplace=True)
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU(inplace=True)
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def normalize_features(mx, norm_row=True):
    """
    Row-normalize sparse matrix
        Code from https://github.com/williamleif/graphsage-simple/
    """

    if norm_row:
        rowsum = np.array(mx.sum(1)) + 0.01
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)

    else:
        column_max = mx.max(dim=0)[0].unsqueeze(0)
        column_min = mx.min(dim=0)[0].unsqueeze(0)
        min_max_column_norm = (mx - column_min) / (column_max - column_min)
        # l2_norm = torch.norm(min_max_column_norm, p=2, dim=-1, keepdim=True)
        mx = min_max_column_norm
    return mx


def apply_non_linearity(tensor, non_linearity, i):
    if non_linearity == 'elu':
        return F.elu(tensor * i - i) + 1
    elif non_linearity == 'relu':
        return F.relu(tensor)
    elif non_linearity == 'none':
        return tensor
    else:
        raise NameError('We dont support the non-linearity yet')


def get_random_mask(features, r, nr):
    nones = torch.sum(features > 0.0).float()
    nzeros = features.shape[0] * features.shape[1] - nones
    # pzeros = nones / nzeros / r * nr
    pzeros = nzeros / nones / r * nr
    probs = torch.zeros(features.shape).to(features.device)
    probs[features == 0.0] = pzeros
    probs[features > 0.0] = 1 / r
    mask = torch.bernoulli(probs)
    return mask


def get_random_mask_ogb(features, r):
    probs = torch.full(features.shape, r)
    mask = torch.bernoulli(probs)
    return mask


def accuracy(preds, labels):
    pred_class = torch.max(preds, 1)[1]
    return torch.sum(torch.eq(pred_class, labels)).float() / labels.shape[0]


def gmean(y_true, y_pred):
    """binary geometric mean of  True Positive Rate (TPR) and True Negative Rate (TNR)

    Args:
            y_true (np.array): label
            y_pred (np.array): prediction
    """

    TP, TN, FP, FN = 0, 0, 0, 0
    for sample_true, sample_pred in zip(y_true, y_pred):
        TP += sample_true * sample_pred
        TN += (1 - sample_true) * (1 - sample_pred)
        FP += (1 - sample_true) * sample_pred
        FN += sample_true * (1 - sample_pred)

    return math.sqrt(TP * TN / (TP + FN) / (TN + FP))


def evaluation_model_prediction(pred_logit, label):
    pred_label = np.argmax(pred_logit, axis=1)
    pred_logit = pred_logit[:, 1]

    accuracy = accuracy_score(label, pred_label)
    f1 = f1_score(label, pred_label, average='macro')
    recall = recall_score(label, pred_label, average='macro')
    auc = roc_auc_score(label, pred_logit)
    ap = average_precision_score(label, pred_logit, average='macro')
    gmean_value = gmean(label, pred_label)

    return Evaluation_Metrics(accuracy=accuracy, macro_F1=f1, recall=recall, auc=auc, ap=ap, gmean=gmean_value)


def nearest_neighbors(X, k, metric, sparse=0):
    adj = kneighbors_graph(X, k, metric=metric)
    sparse_eye = sp.csr_matrix(np.eye(adj.shape[0]))
    adj = adj + sparse_eye
    if not sparse:
        adj = np.array(adj.todense(), dtype=np.float32)
    return adj


def nearest_neighbors_sparse(X, k, metric):
    adj = kneighbors_graph(X, k, metric=metric)
    loop = np.arange(X.shape[0])
    [s_, d_, val] = sp.find(adj)
    s = np.concatenate((s_, loop))
    d = np.concatenate((d_, loop))

    return s, d


def nearest_neighbors_pre_exp(X, k, metric, i):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    adj = adj * i - i
    return adj


def nearest_neighbors_pre_elu(X, k, metric, i):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    adj = adj * i - i
    return adj


def normalize(adj, mode, sparse=False):
    if not sparse:
        if mode == "sym":
            inv_sqrt_degree = 1. / \
                              (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
        elif mode == "row":
            inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EOS)
            return inv_degree[:, None] * adj
        else:
            exit("wrong norm mode")
    else:
        adj = adj.coalesce()
        if mode == "sym":
            inv_sqrt_degree = 1. / \
                              (torch.sqrt(torch.sparse.sum(adj, dim=1).values()))
            D_value = inv_sqrt_degree[adj.indices(
            )[0]] * inv_sqrt_degree[adj.indices()[1]]

        elif mode == "row":
            inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EOS)
            D_value = inv_degree[adj.indices()[0]]
        else:
            exit("wrong norm mode")
        new_values = adj.values() * D_value

        return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size()).coalesce()


def symmetrize(adj):  # only for non-sparse
    return (adj + adj.T) / 2


def top_k_graph_dense(node_embeddings, k, device=None):
    # time1 = time.time()
    raw_graph = torch.mm(node_embeddings, node_embeddings.t())
    values, indices = raw_graph.topk(k=k, dim=-1)
    assert torch.max(indices) < raw_graph.shape[1]
    mask = torch.zeros(raw_graph.shape).to(raw_graph.device)
    mask[torch.arange(raw_graph.shape[0]).view(-1, 1), indices] = 1.
    mask.requires_grad = False
    sparse_graph = raw_graph * mask
    # time2 = time.time()
    # print(f'dense knn time: {time2 - time1} second')
    return sparse_graph


def top_k_graph_based_on_edge_attn(node_embeddings, k, device):
    time1 = time.time()
    node_embeddings = node_embeddings.to(device)
    knn_g = dgl.knn_graph(node_embeddings, k, algorithm='bruteforce-sharemem', dist='cosine', exclude_self=False)
    time2 = time.time()
    # print(f'dgl knn time: {time2 - time1} second')
    return knn_g.edges()


def top_k_graph_dgl(node_embeddings, k, device):
    time1 = time.time()
    node_embeddings = node_embeddings.to(device)
    knn_g = dgl.knn_graph(node_embeddings, k, algorithm='bruteforce-sharemem', dist='cosine')
    edge_v = knn_g.edges()[0]
    edge_u = knn_g.edges()[1]
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    values = cos(node_embeddings[edge_v], node_embeddings[edge_u])
    sparse_coo_tensor = torch.sparse_coo_tensor(torch.vstack(knn_g.edges()), values).to('cpu')

    similarity_graph = sparse_coo_tensor.to_dense()
    time2 = time.time()
    print(f'dgl knn time: {time2 - time1} second')
    return similarity_graph


def similarity2g(raw_graph, K):
    values, indices = raw_graph.topk(k=int(K), dim=-1)
    row_index = torch.LongTensor(np.arange(indices.shape[0])).unsqueeze(-1)
    row_index = row_index.repeat(1, int(K))
    g = dgl.graph((torch.flatten(row_index), torch.flatten(indices)))
    g.edata['w'] = torch.flatten(values)
    return g


def knn_fast(X, k, b):
    X = F.normalize(X, dim=1, p=2)
    index = 0
    values = torch.zeros(X.shape[0] * (k + 1)).to(X.device)
    rows = torch.zeros(X.shape[0] * (k + 1)).to(X.device)
    cols = torch.zeros(X.shape[0] * (k + 1)).to(X.device)
    norm_row = torch.zeros(X.shape[0]).to(X.device)
    norm_col = torch.zeros(X.shape[0]).to(X.device)
    while index < X.shape[0]:
        if (index + b) > (X.shape[0]):
            end = X.shape[0]
        else:
            end = index + b
        sub_tensor = X[index:index + b]
        similarities = torch.mm(sub_tensor, X.t())
        vals, inds = similarities.topk(k=k + 1, dim=-1)
        values[index * (k + 1):(end) * (k + 1)] = vals.view(-1)
        cols[index * (k + 1):(end) * (k + 1)] = inds.view(-1)
        rows[index * (k + 1):(end) * (k + 1)] = torch.arange(index,
                                                             end).view(-1, 1).repeat(1, k + 1).view(-1)
        norm_row[index: end] = torch.sum(vals, dim=1)
        norm_col.index_add_(-1, inds.view(-1), vals.view(-1))
        index += b
    norm = norm_row + norm_col
    rows = rows.long()
    cols = cols.long()
    values *= (torch.pow(norm[rows], -0.5) * torch.pow(norm[cols], -0.5))
    return rows, cols, values


def sp_sparse2torch_sparse(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def gen_dgl_graph(index1, index2, edge_w=None, ndata=None):
    g = dgl.graph((index1, index2))
    if edge_w is not None:
        g.edata['w'] = edge_w
    if ndata is not None:
        g.ndata['h'] = ndata
    return g

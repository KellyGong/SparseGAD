import dgl
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import dgl.function as fn
from utils import create_activation, top_k_graph_based_on_edge_attn, gen_dgl_graph, normalize

from torch_geometric.utils import sort_edge_index


class GPR_ATT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj, k, threshold, device):
        super(GPR_ATT, self).__init__()

        self.k = k
        self.device = device
        self.threshold = threshold
        self.inlinear = nn.Linear(in_channels, hidden_channels)
        self.outlinear = nn.Linear(hidden_channels, out_channels)

        torch.nn.init.xavier_uniform_(self.inlinear.weight)
        torch.nn.init.xavier_uniform_(self.outlinear.weight)

        self.gnn = GPR_sparse(hidden_channels, num_layers, dropout, dropout_adj)
        self.extractor = ExtractorMLP(hidden_channels)

    def forward(self, x, adj=None, g=None):
        with g.local_scope():
            h = self.inlinear(x)
            h_gnn = self.gnn.forward(h, adj, g)
            g.edata['attn'] = self.extractor(h_gnn, g.edges())
            h_gnn = self.gnn.forward(h, adj, g, edge_attn=True)
            x = self.outlinear(h_gnn)
        return x

    def sparsify(self, g, Adj):
        x_h = self.gen_node_emb(g.ndata['h'], Adj, g)
        new_edges = top_k_graph_based_on_edge_attn(x_h, k=self.k, device=self.device)
        edge_attn = torch.abs(self.gen_edge_attn(g.ndata['h'], Adj, g))
        filtered_edge = (g.edges()[0][edge_attn > self.threshold], g.edges()[1][edge_attn > self.threshold])
        new_g = gen_dgl_graph(torch.cat((filtered_edge[0], new_edges[0])), torch.cat((filtered_edge[1], new_edges[1])),
                              ndata=g.ndata['h']).to('cpu')
        new_g = dgl.to_simple(new_g)
        Adj = normalize(new_g.adj(), 'sym', 1)
        new_g = gen_dgl_graph(Adj.indices()[0], Adj.indices()[1], Adj.values(), g.ndata['h'].to('cpu'))
        Adj = new_g.adj()
        new_g = new_g.to(self.device)
        return new_g, Adj

    def gen_node_emb(self, x, adj=None, g=None):
        with g.local_scope():
            h = self.inlinear(x)
            h_gnn = self.gnn.forward(h, adj, g)
            h_gnn = self.extractor.feature_extractor(h_gnn)
            return h_gnn

    def gen_edge_attn(self, x, adj=None, g=None):
        with g.local_scope():
            h = self.inlinear(x)
            h_gnn = self.gnn.forward(h, adj, g)
            return self.extractor(h_gnn, g.edges())

    @staticmethod
    def sampling(att_log_logit, training, temp=1):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern

    @staticmethod
    def get_r(decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att


class ExtractorMLP(nn.Module):
    def __init__(self, hidden_size, activation='relu', dropout=0.2):
        super(ExtractorMLP, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(p=dropout),
            create_activation(activation),
            nn.Linear(hidden_size, hidden_size),
        )
        self.cos = nn.CosineSimilarity(dim=1)
        self._init_weight(self.feature_extractor)

    @staticmethod
    def _init_weight(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)

    @staticmethod
    def reorder_like(from_edge_index, to_edge_index, values):
        from_edge_index, values = sort_edge_index(from_edge_index, values)
        ranking_score = to_edge_index[0] * (to_edge_index.max() + 1) + to_edge_index[1]
        ranking = ranking_score.argsort().argsort()
        if not (from_edge_index[:, ranking] == to_edge_index).all():
            raise ValueError("Edges in from_edge_index and to_edge_index are different, impossible to match both.")
        return values[ranking]

    def symmetric(self, edge_index, attn_logits):
        row, col = edge_index
        trans_attn_logits = self.reorder_like(torch.stack(edge_index), torch.stack((col, row)), attn_logits)
        edge_attn = (trans_attn_logits + attn_logits) / 2
        return edge_attn

    def forward(self, emb, edge_index, batch=None):
        col, row = edge_index
        f1, f2 = emb[col], emb[row]
        attn_logits = self.cos(self.feature_extractor(f1), self.feature_extractor(f2))
        return attn_logits


class GCNConv_dgl_attn(nn.Module):
    def __init__(self, input_size, output_size):
        super(GCNConv_dgl_attn, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x, g):
        g.ndata['h'] = self.linear(x)
        g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum(msg='m', out='h'))
        return g.ndata['h']


class GPR_sparse(nn.Module):
    def __init__(self, hidden_channels, num_layers, dropout, dropout_adj):
        super(GPR_sparse, self).__init__()

        self.layers = nn.ModuleList([GCNConv_dgl_attn(hidden_channels, hidden_channels) for _ in range(num_layers)])
        # GPR temprature initialize
        alpha = 0.1
        temp = alpha * (1 - alpha) ** np.arange(num_layers + 1)
        temp[-1] = (1 - alpha) ** num_layers
        self.temp = nn.Parameter(torch.from_numpy(temp))

        self.dropout = dropout
        self.dropout_adj_p = dropout_adj

    def forward(self, x, adj=None, g=None, edge_attn=False):
        if edge_attn:
            g.edata['w'] = g.edata['w'] * g.edata['attn']
        g.edata['w'] = F.dropout(g.edata['w'], p=self.dropout_adj_p, training=self.training)
        hidden = x * self.temp[0]
        for i, conv in enumerate(self.layers):
            x = conv(x, g)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            hidden += x * self.temp[i + 1]
        return hidden

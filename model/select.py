from .gpr_att import GPR_ATT


def get_gnn_model(model_str, in_channels, hidden_channels, out_channels, num_layers,
                  dropout, dropout_adj, k, threshold, device):
    if model_str == 'gpr_att':
        Model = GPR_ATT
    else:
        raise NotImplementedError

    return Model(in_channels=in_channels,
                 hidden_channels=hidden_channels,
                 out_channels=out_channels,
                 num_layers=num_layers,
                 dropout=dropout,
                 dropout_adj=dropout_adj,
                 k=k,
                 threshold=threshold,
                 device=device)

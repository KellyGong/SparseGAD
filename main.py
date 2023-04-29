from utils import *
from model import get_gnn_model
from data_loader import load_data
import torch.nn.functional as F
import torch
import numpy as np
import copy
import argparse

torch.set_num_threads(8)

EOS = 1e-10

setup_seed(2022)


class Experiment:
    def __init__(self, args):
        super(Experiment, self).__init__()
        self.args = args
        self.device = args.device

    def get_classification_loss(self, model, mask, features, labels, Adj=None, g=None):
        logits = model(features, Adj, g)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        eval_res = evaluation_model_prediction(
            logp[mask].detach().cpu().numpy(), labels[mask].cpu().numpy())
        return loss, eval_res

    def get_loss_masked_features(self, model, features, mask, ogb, noise, loss_t):
        if ogb:
            if noise == 'mask':
                masked_features = features * (1 - mask)
            elif noise == "normal":
                noise = torch.normal(
                    0.0, 1.0, size=features.shape).to(features.device)
                masked_features = features + (noise * mask)

            logits, Adj, g = model(features, masked_features)
            indices = mask > 0

            if loss_t == 'bce':
                features_sign = torch.sign(features) * 0.5 + 0.5
                loss = F.binary_cross_entropy_with_logits(
                    logits[indices], features_sign[indices], reduction='mean')
            elif loss_t == 'mse':
                loss = F.mse_loss(
                    logits[indices], features[indices], reduction='mean')
        else:
            masked_features = features * (1 - mask)
            logits, Adj, g = model(features, masked_features)
            indices = mask > 0
            loss = F.binary_cross_entropy_with_logits(
                logits[indices], features[indices], reduction='mean')
        return loss, Adj, g

    def train_classification_gcn(self, Adj, features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, args,
                                 g=None):
        model = get_gnn_model(model_str=args.model, in_channels=nfeats, hidden_channels=args.hidden,
                              out_channels=nclasses, num_layers=args.nlayers,
                              dropout=args.dropout2, dropout_adj=args.dropout_adj2, sparse=args.sparse)

        bad_counter = 0
        best_val = None
        best_model = None
        best_loss = 0
        best_train_loss = 0

        model = model.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        test_mask = test_mask.to(self.device)
        features = features.to(self.device)
        labels = labels.to(self.device)
        g = g.to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.w_decay)

        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad()
            loss, train_res = self.get_classification_loss(
                model, train_mask, features, labels, Adj, g)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                val_loss, val_res = self.get_classification_loss(
                    model, val_mask, features, labels, Adj, g)

                if best_val is None or val_res.auc > best_val.auc:
                    bad_counter = 0
                    best_val = val_res
                    best_model_weight = {k: v.cpu() for k, v in copy.deepcopy(model.state_dict()).items()}
                    best_loss = val_loss
                    best_train_loss = loss
                    print("Epoch {} Val Loss {:.4f}, Val Auc {:.4f}, Val AP {:.4f}, Val macro_F1 {:.4F}".format(
                        epoch, best_loss, best_val.auc, best_val.ap, best_val.macro_F1))
                else:
                    bad_counter += 1

                if bad_counter >= args.patience:
                    break

        print("Val Loss {:.4f}, Val Auc {:.4f}, Val AP {:.4f}, Val macro_F1 {:.4F}".format(
            best_loss, best_val.auc, best_val.ap, best_val.macro_F1))
        with torch.no_grad():
            model.eval()
            model.load_state_dict(best_model_weight)
            test_loss, test_res = self.get_classification_loss(
                model, test_mask, features, labels, Adj, g)
            print("Test Loss {:.4f}, Test Auc {:.4f}, Test AP {:.4f}, Test macro_F1 {:.4F}".format(
                test_loss, test_res.auc, test_res.ap, test_res.macro_F1))
            torch.save(model, 'model.pt')
        return best_val, test_res, model

    def valid_classification_gcn(self, Adj, features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, args,
                                 g=None):
        model = self.load_model()
        model = model.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        test_mask = test_mask.to(self.device)
        features = features.to(self.device)
        labels = labels.to(self.device)
        g = g.to(self.device)
        val_loss, val_res = self.get_classification_loss(
            model, val_mask, features, labels, Adj, g)
        test_loss, test_res = self.get_classification_loss(
            model, test_mask, features, labels, Adj, g)

        print("Test Loss {:.4f}, Test Auc {:.4f}, Test AP {:.4f}, Test macro_F1 {:.4F}".format(
            test_loss, test_res.auc, test_res.ap, test_res.macro_F1))
        return val_res, test_res, model

    @staticmethod
    def load_model(path='model.pt', device='cpu'):
        model = torch.load(path, map_location=device)
        return model

    def train_attn_gnn(self, args):
        features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, g = load_data(args)

        test_results = []
        validation_results = []

        Adj = normalize(g.adj(), args.normalization, args.sparse)

        if args.sparse:
            g = gen_dgl_graph(Adj.indices()[0], Adj.indices()[
                1], Adj.values(), features)
            Adj = g.adj()

        update_adj_epoch = 200

        for trial in range(args.ntrials):
            model = get_gnn_model(model_str=args.model, in_channels=nfeats, hidden_channels=args.hidden,
                                  out_channels=nclasses,
                                  num_layers=args.nlayers, dropout=args.dropout2, dropout_adj=args.dropout_adj2,
                                  k=args.k, threshold=args.threshold, device=args.device)

            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=args.w_decay)

            model = model.to(self.device)
            train_mask = train_mask.to(self.device)
            val_mask = val_mask.to(self.device)
            test_mask = test_mask.to(self.device)
            # features = features.to(self.device)
            labels = labels.to(self.device)
            g = g.to(self.device)

            best_val = None

            for epoch in range(1, args.epochs + 1):
                model.train()
                optimizer.zero_grad()

                loss, train_res = self.get_classification_loss(
                    model, train_mask[trial], g.ndata['h'], labels, Adj, g)

                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    model.eval()
                    val_loss, val_res = self.get_classification_loss(
                        model, val_mask[trial], g.ndata['h'], labels, Adj, g)
                    if best_val is None or val_res.auc > best_val.auc:
                        best_val = val_res
                        print("epoch {}, Val Loss {:.4f}, Val Auc {:.4f}, Val AP {:.4f}, Val macro_F1 {:.4F}".format(
                            epoch, val_loss, best_val.auc, best_val.ap, best_val.macro_F1))
                        test_loss_, test_res = self.get_classification_loss(model, test_mask[trial], g.ndata['h'],
                                                                            labels,
                                                                            Adj, g)
                        print(
                            "epoch {}, Test Loss {:.4f}, Test Auc {:.4f}, Test AP {:.4f}, Test macro_F1 {:.4F}".format(
                                epoch, test_loss_, test_res.auc, test_res.ap, test_res.macro_F1))

                if epoch % update_adj_epoch == 0:
                    x_h = model.gen_node_emb(g.ndata['h'], Adj, g)
                    new_edges = top_k_graph_based_on_edge_attn(x_h, k=args.k, device=self.device)
                    g = gen_dgl_graph(torch.cat((g.edges()[0], new_edges[0])),
                                      torch.cat((g.edges()[1], new_edges[1])),
                                      ndata=g.ndata['h']).to('cpu')
                    g = dgl.to_simple(g)
                    Adj = normalize(g.adj(), args.normalization, args.sparse)
                    g = gen_dgl_graph(Adj.indices()[0], Adj.indices()[1], Adj.values(), g.ndata['h'].to('cpu'))
                    Adj = g.adj()
                    g = g.to(self.device)

            validation_results.append(best_val)
            test_results.append(test_res)

            print("Test Auc {:.4f}, Test AP {:.4f}, Test macro_F1 {:.4F}".format(test_res.auc, test_res.ap,
                                                                                 test_res.macro_F1))

        self.print_results(validation_results, test_results)
        return test_results

    def print_results(self, val_results, test_results):
        valid_aucs, test_aucs = [val_res.auc for val_res in val_results], [
            test_res.auc for test_res in test_results]
        valid_ap, test_ap = [val_res.ap for val_res in val_results], [
            test_res.ap for test_res in test_results]
        valid_macro_f1, test_macro_f1 = [val_res.macro_F1 for val_res in val_results], [
            test_res.macro_F1 for test_res in test_results]
        valid_gmean, test_gmean = [val_res.gmean for val_res in val_results], [
            test_res.gmean for test_res in test_results]

        print(
            f"mean+-std of valid auc: {np.mean(valid_aucs):.4f}+-{np.std(valid_aucs):.4f}, test auc: {np.mean(test_aucs):.4f}+-{np.std(test_aucs):.4f}")
        print(
            f"mean+-std of valid ap: {np.mean(valid_ap):.4f}+-{np.std(valid_ap):.4f}, test ap: {np.mean(test_ap):.4f}+-{np.std(test_ap):.4f}")
        print(
            f"mean+-std of valid macro f1: {np.mean(valid_macro_f1):.4f}+-{np.std(valid_macro_f1):.4f}, test macro f1: {np.mean(test_macro_f1):.4f}+-{np.std(test_macro_f1):.4f}")
        print(
            f"mean+-std of valid gmean: {np.mean(valid_gmean):.4f}+-{np.std(valid_gmean):.4f}, test gmean: {np.mean(test_gmean):.4f}+-{np.std(test_gmean):.4f}")


def print_test_statics(test_results):
    test_aucs = [test_res.auc for test_res in test_results]
    test_ap = [test_res.ap for test_res in test_results]
    test_macro_f1 = [test_res.macro_F1 for test_res in test_results]

    print(
        f"mean+-std of test auc: {np.mean(test_aucs):.4f}+-{np.std(test_aucs):.4f}, ap: {np.mean(test_ap):.4f}+-{np.std(test_ap):.4f}, test macro f1: {np.mean(test_macro_f1):.4f}+-{np.std(test_macro_f1):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Initial learning rate.')
    parser.add_argument('--w_decay', type=float, default=0.0005,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=32,
                        help='Number of hidden units.')
    parser.add_argument('--dropout2', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dropout_adj2', type=float, default=0.25,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--nlayers', type=int, default=2, help='#layers')
    parser.add_argument('--patience', type=int, default=50,
                        help='Patience for early stopping')
    parser.add_argument('--ntrials', type=int, default=1,
                        help='Number of trials')
    parser.add_argument('--k', type=int, default=5,
                        help='k for initializing with knn')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='filter edge with lowest edge attention')
    parser.add_argument('--train_ratio', type=float, default=0.4)
    parser.add_argument('--knn_metric', type=str, default='cosine',
                        help='See choices', choices=['cosine'])
    parser.add_argument('--normalization', type=str, default='sym')
    parser.add_argument('--sparse', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='yelp', help='See choices',
                        choices=['amazon', 'yelp', 'reddit'])
    parser.add_argument('--mode', type=str, default="attn_gnn", help='See choices',
                        choices=['attn_gnn'])
    parser.add_argument('--model', type=str, default="gpr_att", help='See choices',
                        choices=['gpr_att'])
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args()

    print(args)

    experiment = Experiment(args)

    experiment.train_attn_gnn(args)

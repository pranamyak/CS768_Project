import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""
from layers.gcn_layer import GCNLayer
from layers.mlp_readout_layer import MLPReadout
import pdb
import torch.autograd as autograd
import numpy as np


class HRMNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.gcn_net = GCNNet(net_params)
        self.embedding_h = self.gcn_net.embedding_h
        self.layers = self.gcn_net.layers
        self.MLP_layer = self.gcn_net.MLP_layer
        self.irm_lambda = net_params['irm_lambda']
        #self.irm_penalty_anneal_iters = net_params['irm_penalty_anneal_iters']
        self.register_buffer("q", torch.Tensor())
        random_state = np.random.RandomState(10)

        self.groupdro_eta = 1e-1
    def irm_penalty(self, losses):
        grad_1 = autograd.grad(losses[0::2].mean(), [self.scale], create_graph=True)[0]
        grad_2 = autograd.grad(losses[1::2].mean(), [self.scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss
    '''
    def forward(self, batch_graphs, batch_x, batch_labels, envs, data_mask, device):
        avg_loss = 0.
        penalty = 0.
        n_groups_per_batch = len(envs)
        if len(self.q) == 0:
            self.q = torch.ones(len(envs)).to(device)

        losses = torch.zeros(len(envs)).to(device)

        results = self.gcn_net(batch_graphs, batch_x, None, None)
        grad_list = []
        grad_avg = 0.0
        num_envs = len(envs)
        all_results = []
        
        loss_avg = 0.0
        for m, env in enumerate(envs):
            g = env[0]
            h = env[1]
            labels = env[2]
            indices = env[3]
        #    results = self.gcn_net(g, h, None, None)
            #all_graphs += g
            #all_results.append(results)
            loss = self.loss(results[indices], batch_labels[indices])
            loss_avg += loss/num_envs
            grad_single = autograd.grad(loss, self.gcn_net.parameters(), create_graph=True)[0].reshape(-1)
            grad_avg += grad_single / num_envs
            grad_list.append(grad_single)
            #all_labels.append(labels)
            #all_group_loss.append(group_loss)
        #    print("label*****************", labels)

        #results = torch.cat(all_results, axis=0)
        penalty = torch.tensor(np.zeros(grad_single.shape, dtype=np.float32)).to(device)
        for gradient in grad_list:
            penalty += (gradient - grad_avg)**2
        penalty_detach = penalty.mean()
        return loss_avg, results

    '''

    def forward(self, batch_graphs, batch_x, batch_labels, envs, data_mask, epoch, device):
        avg_loss = 0.
        penalty = 0.
        n_groups_per_batch = len(envs)
        if len(self.q) == 0:
            self.q = torch.ones(len(envs)).to(device)

        losses = torch.zeros(len(envs)).to(device)

        results = self.gcn_net(batch_graphs, batch_x, None, data_mask)
        grad_list = []
        grad_avg = 0.0
        num_envs = len(envs)
        all_results = []
        
        loss_avg = 0.0
        all_labels = []
        for m, env in enumerate(envs):
            g = env[0]
            h = env[1]
            labels = env[2]
            indices = env[3]
        #    results = self.gcn_net(g, h, None, None)
            #all_graphs += g
       #     all_results.append(results)
       #     all_labels.append(labels)
            loss = self.loss(results[indices], labels)
            loss_avg += loss/num_envs
            grad_single = autograd.grad(loss, self.gcn_net.parameters(), create_graph=True)[0].reshape(-1)
            grad_avg += grad_single / num_envs
            grad_list.append(grad_single)
            #all_labels.append(labels)
            #all_group_loss.append(group_loss)
        #    print("label*****************", labels)

        #results = torch.cat(all_results, axis=0)
        #labels = torch.cat(all_labels, axis=0)
        penalty = torch.tensor(np.zeros(grad_single.shape, dtype=np.float32)).to(device)
        for gradient in grad_list:
            penalty += (gradient - grad_avg)**2
        penalty_detach = penalty.mean()
        irm_lambda = self.irm_lambda if epoch > 100 else 1.0
        return loss_avg + self.irm_lambda * penalty_detach, results, batch_labels

    def test(self, g, h, e, data_mask=None):
        results = self.gcn_net(g, h, None, data_mask)            
        return results

class GroupDRONet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.gcn_net = GCNNet(net_params)
        self.embedding_h = self.gcn_net.embedding_h
        self.layers = self.gcn_net.layers
        self.MLP_layer = self.gcn_net.MLP_layer
        #self.irm_lambda = net_params['irm_lambda']
        #self.irm_penalty_anneal_iters = net_params['irm_penalty_anneal_iters']
        self.register_buffer("q", torch.Tensor())
        random_state = np.random.RandomState(10)

        self.groupdro_eta = 1e-1
    def irm_penalty(self, losses):
        grad_1 = autograd.grad(losses[0::2].mean(), [self.scale], create_graph=True)[0]
        grad_2 = autograd.grad(losses[1::2].mean(), [self.scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss(reduction='none')
        loss = criterion(pred, label)
        return loss
    def forward(self, batch_graphs, batch_x, batch_labels, envs, data_mask, device):
        avg_loss = 0.
        penalty = 0.
        n_groups_per_batch = len(envs)
        if len(self.q) == 0:
            self.q = torch.ones(len(envs)).to(device)

        losses = torch.zeros(len(envs)).to(device)

        results = self.gcn_net(batch_graphs, batch_x, None, data_mask)
        for m, env in enumerate(envs):
            #g = env[0]
            #h = env[1]
            #labels = env[2]
            #results = self.gcn_net(g, h, None, mask)
            #all_graphs += g
            #all_results.append(results)
            if not len(env):
                continue 
            losses[m] = F.cross_entropy(results[env], batch_labels[env])
            self.q[m] *= (self.groupdro_eta * losses[m].data).exp()
            #all_labels.append(labels)
            #all_group_loss.append(group_loss)
        #    print("label*****************", labels)
 
        self.q /= self.q.sum()
        loss = torch.dot(losses, self.q)
        return loss, results

    def test(self, g, h, e, data_mask=None):
        results = self.gcn_net(g, h, None, None, None)
        return results

class IRMNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.gcn_net = GCNNet(net_params)
        self.embedding_h = self.gcn_net.embedding_h
        self.layers = self.gcn_net.layers
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        self.MLP_layer = nn.Linear(out_dim*2, n_classes)        
        #self.irm_lambda = net_params['irm_lambda']
        #self.irm_penalty_anneal_iters = net_params['irm_penalty_anneal_iters']
        self.scale = torch.tensor(1.).cuda().requires_grad_()
        #for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight)
        #         nn.init.constant_(m.bias, 0)
    def irm_penalty(self, losses):
        grad_1 = autograd.grad(losses[0::2].mean(), [self.scale], create_graph=True)[0]
        grad_2 = autograd.grad(losses[1::2].mean(), [self.scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss(reduction='none')
        loss = criterion(pred, label)
        return loss
    def forward(self, batch_graphs, batch_x, batch_labels, envs, data_mask, data_mask_node, device):
        avg_loss = 0.
        penalty = 0.
        n_groups_per_batch = len(envs)
        all_results = []
        all_group_loss = []
        all_labels = []
        all_graphs = []

        results = self.gcn_net(batch_graphs, batch_x, None, None, None)
        #print("penalty", avg_loss,penalty)
        return results

    def test(self, g, h, e, data_mask=None, data_mask_node=None):
        results = self.gcn_net(g, h, None, None, None)
        return results

class GCNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([GCNLayer(hidden_dim, hidden_dim, F.relu, dropout,
                                              self.batch_norm, self.residual) for _ in range(n_layers-1)])
        self.layers.append(GCNLayer(hidden_dim, out_dim, F.relu, dropout, self.batch_norm, self.residual))
        #self.MLP_layer = MLPReadout(out_dim, n_classes)        

        #self.mlp = nn.Linear(out_dim, n_classes)
    def forward(self, g, h, e, data_mask=None, data_mask_node=None):

        if data_mask_node is not None:
            h = h * data_mask_node
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        conv = self.layers[0]
        #h1 = conv(g, h, data_mask=data_mask, data_mask_node=None)
        #conv1 = self.layers[1]
        #h2 = conv1(g, h1, data_mask=data_mask, data_mask_node=None)
        for conv in self.layers:
            h = conv(g, h, data_mask=data_mask, data_mask_node=None)
        g.ndata['h'] = h
        
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            
        return hg
    
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss





class GCNMasker(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = 2
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([GCNLayer(hidden_dim, hidden_dim, F.relu, dropout,
                                              self.batch_norm, self.residual) for _ in range(n_layers-1)])
        self.layers.append(GCNLayer(hidden_dim, out_dim, F.relu, dropout, self.batch_norm, self.residual))

        self.sigmoid = nn.Sigmoid()    
        self.mlp = MLPReadout(hidden_dim * 2, 1)
        self.node_mlp = MLPReadout(hidden_dim, 1)
        #for m in self.modules():
        #    print("m out", m, isinstance(m, nn.Linear))
       #     if isinstance(m, nn.Linear):
       #         print("m", m)
       #         nn.init.xavier_uniform_(m.weight)
       #         nn.init.constant_(m.bias, 0)
    def forward(self, g, h, e):
        
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        #for conv in self.layers:
        #    h = conv(g, h)
        g.ndata['h'] = h
        node_score = self.node_score(g)
        link_score = self.concat_mlp_score(g)
        return link_score, node_score
    
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss
    
    def inner_product_score(self, g):
        
        row, col = g.edges()
        link_score = torch.sum(g.ndata['h'][row] * g.ndata['h'][col], dim=1)
        link_score = self.sigmoid(link_score)
        return link_score

    def node_score(self, g):

        link_score = self.node_mlp(g.ndata['h'])
        link_score = self.sigmoid(link_score)
        
        return link_score


    def concat_mlp_score(self, g):
        
        row, col = g.edges()
        link_score = torch.cat((g.ndata['h'][row], g.ndata['h'][col]), dim=1)
        link_score = self.mlp(link_score)
        link_score = self.sigmoid(link_score)
        
        return link_score



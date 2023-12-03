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

import scipy.sparse as sp



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
        self.in_feat_dropout = nn.Dropout(in_feat_dropout, inplace=True)
        
        self.layers = nn.ModuleList([GCNLayer(hidden_dim, hidden_dim, F.relu_, dropout,
                                              self.batch_norm, self.residual) for _ in range(n_layers-1)])
        self.layers.append(GCNLayer(hidden_dim, out_dim, F.relu_, dropout, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)        

        #self.mlp = nn.Linear(out_dim, n_classes)
        #for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_normal_(m.weight)
       #          nn.init.constant_(m.bias, 0)

    def forward(self, g, h, e, data_mask=None, data_mask_node=None):
        if data_mask_node is not None:
            h = h * data_mask_node
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        conv = self.layers[0]
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
            
        return self.MLP_layer(hg)
    
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
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout, inplace=True)
        
        self.layers = nn.ModuleList([GCNLayer(hidden_dim, hidden_dim, F.relu, dropout,
                                              self.batch_norm, self.residual) for _ in range(n_layers-1)])
        self.layers.append(GCNLayer(hidden_dim, out_dim, F.relu, dropout, self.batch_norm, self.residual))

        self.sigmoid = nn.Sigmoid()    
        self.mlp = nn.Linear(hidden_dim * 2, 1)
        self.node_mlp = nn.Linear(hidden_dim, 1)
    def forward(self, g, h, e):
        
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        for conv in self.layers:
            h = conv(g, h)
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



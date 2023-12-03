import torch
import torch.nn as nn
import math
import time
import pruning
from train.metrics import accuracy_MNIST_CIFAR as accuracy
import pdb
import numpy as np
import dgl
from utils import GeneralizedCELoss
import pickle

class DGLFormDataset(torch.utils.data.Dataset):
    """
        DGLFormDataset wrapping graph list and label list as per pytorch Dataset.
        *lists (list): lists of 'graphs' and 'labels' with same len().
    """
    def __init__(self, *lists):
        assert all(len(lists[0]) == len(li) for li in lists)
        self.lists = lists
        self.graph_lists = lists[0]
        self.vector_c = lists[1]
        self.vector_b = lists[2]
        self.graph_labels = lists[3]
        self.bias_label = lists[4]

    def __getitem__(self, index):
        return tuple(li[index] for li in self.lists)

    def __len__(self):
        return len(self.lists[0])
 
bias_criterion = GeneralizedCELoss(q=0.7)
criterion = nn.BCELoss() 




def train_model_and_masker(model, optimizer,  device, data_loader, epoch, args):

    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
        batch_graphs = dgl.batch(batch_graphs).to(device)

        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        #vector_c = vector_c.to(device)

        #vector_b = vector_b.to(device)
        #batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.to(device)

        # Train generator
        optimizer.zero_grad()
        #label.fill_(real_label)

        batch_scores = model(batch_graphs, batch_x, None, None, None)
        # Calculate G's loss based on this outpuii
        # Calculate gradients for G
        loss = model.loss(batch_scores, batch_labels)
        loss.backward()

        optimizer.step()


        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)
        
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data

    return epoch_loss, epoch_train_acc, optimizer

         

    

def eval_acc_with_mask(model, device, data_loader, epoch, binary=False, val=False):

    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            batch_graphs = dgl.batch(batch_graphs).to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            #batch_e = batch_graphs.edata['feat'].to(device)
            batch_e = None
            batch_labels = batch_labels.to(device)

            batch_scores = model(batch_graphs, batch_x, batch_e, None, None)

            loss = model.loss(batch_scores, batch_labels)
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data
        
    return epoch_test_loss, epoch_test_acc



def train_epoch(model, optimizer, device, data_loader, epoch, args):

    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_labels, batch_bias_labels) in enumerate(data_loader):
        batch_graphs = dgl.batch(batch_graphs).to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        #batch_e = batch_graphs.edata['feat'].to(device)
        batch_e = None
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        
        batch_scores = model.forward(batch_graphs, batch_x, batch_e)
        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)
    
        if iter % 40 == 0:
            print('-'*120)
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
                    'Epoch: [{}/{}]  Iter: [{}/{}]  Loss: [{:.4f}]'
                    .format(epoch + 1, args.eval_epochs, iter, len(data_loader), epoch_loss / (iter + 1), epoch_train_acc / nb_data * 100))
    
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data

    return epoch_loss, epoch_train_acc, optimizer



def evaluate_network(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels, batch_bias_labels) in enumerate(data_loader):
            batch_graphs = dgl.batch(batch_graphs).to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            #batch_e = batch_graphs.edata['feat'].to(device)
            batch_e = None
            batch_labels = batch_labels.to(device)
            
            batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            loss = model.loss(batch_scores, batch_labels) 
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data
        
    return epoch_test_loss, epoch_test_acc

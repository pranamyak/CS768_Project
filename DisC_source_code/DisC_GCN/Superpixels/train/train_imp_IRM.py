import torch
import torch.nn as nn
import math
import time
import pruning
from train.metrics import accuracy_MNIST_CIFAR as accuracy
import pdb
import numpy as np
import dgl
def group_envs(batch_graphs, batch_labels, batch_bias_labels, device):
    bias_unique = torch.unique(batch_bias_labels)
    envs = []
    for bias in bias_unique:
        indices = np.where((batch_bias_labels == bias).numpy())[0]
        indices = torch.LongTensor(indices)
        tmp_graphs = [batch_graphs[ind] for ind in indices]
        graphs = dgl.batch(tmp_graphs).to(device)
        env_batch_labels = batch_labels[indices].to(device)
        
        batch_x = graphs.ndata['feat'].to(device)  # num x feat
        #print("bias", bias, env_batch_labels)
        envs.append([dgl.unbatch(graphs), batch_x, env_batch_labels])
    return envs
    



def train_model_and_masker(model, masker, optimizer, device, data_loader, epoch, args):

    model.train()
    masker.train()
    
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    mask_distribution = []
    for iter, (batch_graphs, batch_labels, batch_bias_labels) in enumerate(data_loader):
        envs = group_envs(batch_graphs, batch_labels, batch_bias_labels, device)
        batch_graphs = dgl.batch(batch_graphs).to(device)

        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        #batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        data_maskers = []
        for env in envs: 
            data_mask = masker(dgl.batch(env[0]), env[1], None)
            data_maskers.append(data_mask)
            mask_dis = pruning.plot_mask(data_mask)
            mask_distribution.append(mask_dis)
        loss, batch_scores = model(batch_graphs, batch_x, data_maskers, device)
        #loss = model.loss(batch_scores, batch_labels).mean()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        print("batch_scores shape:", batch_scores.shape)
        epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)
    
        if iter % 40 == 0:
            print('-'*120)
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' +
                    'Epoch: [{}/{}] Iter: [{}/{}] Loss:[{:.4f}] Train Acc:[{:.2f}] LR1:[{:.6f}] LR2:[{:.6f}] | 0-0.2:[{:.2f}%] 0.2-0.4:[{:.2f}%] 0.4-0.6:[{:.2f}%] 0.6-0.8:[{:.2f}%] 0.8-1.0:[{:.2f}%]'
                    .format(epoch + 1, 
                            args.mask_epochs, 
                            iter, 
                            len(data_loader), 
                            epoch_loss / (iter + 1), 
                            epoch_train_acc / nb_data * 100,
                            optimizer.param_groups[0]['lr'],
                            optimizer.param_groups[1]['lr'],
                            mask_dis[0] * 100,
                            mask_dis[1] * 100,
                            mask_dis[2] * 100,
                            mask_dis[3] * 100,
                            mask_dis[4] * 100))

    mask_distribution = torch.tensor(mask_distribution).mean(dim=0)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data

    return epoch_loss, epoch_train_acc, optimizer, mask_distribution

         

    

def eval_acc_with_mask(model, masker, device, data_loader, epoch, binary=False):

    model.eval()
    masker.eval()
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
            
            data_mask = masker(batch_graphs, batch_x, batch_e)
            if binary:
                data_mask = pruning.binary_mask(data_mask, args.pa)

            batch_scores = model.test(batch_graphs, batch_x, batch_e, data_mask)
            loss = model.loss(batch_scores, batch_labels).mean() 
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

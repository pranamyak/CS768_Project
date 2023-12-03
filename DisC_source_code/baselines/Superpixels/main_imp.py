import numpy as np
import os
import time
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from train.train_imp import train_model_and_masker, eval_acc_with_mask, train_epoch, evaluate_network
from nets.superpixels_graph_classification.load_net import gnn_model, mask_model, GCNNet


from data.data import LoadData # import dataset
import pruning
import copy
import pdb

batch_size=256
#torch.autograd.set_detect_anomaly(True)
def train_get_mask(dataset_ori, net_params, things_dict, imp_num, filename, args):
    t0 = time.time()
    print("process ...")
    #trainset_ori, valset_ori, testset_ori = dataset_ori.train, dataset_ori.val, dataset_ori.test
    
    trainset_ori, valset_ori, biased_testset_ori, unbiased_testset_ori = dataset_ori.train, dataset_ori.val, dataset_ori.biased_test, dataset_ori.unbiased_test
    train_loader_ori = DataLoader(trainset_ori, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=dataset_ori.train_collate)
    val_loader_ori = DataLoader(valset_ori, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=dataset_ori.collate)
    biased_test_loader_ori = DataLoader(biased_testset_ori, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=dataset_ori.collate)
    unbiased_test_loader_ori = DataLoader(unbiased_testset_ori, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=dataset_ori.collate)

    device = torch.device("cuda")
    model = gnn_model(net_params['model'], net_params).to(device)
    
    train_label = [train[1] for train in trainset_ori]
    if things_dict is not None:

        trainset_pru, valset_pru, biased_testset_pru, unbiased_testset_pru = things_dict['trainset_pru'], things_dict['valset_pru'], things_dict['biased_testset_pru'], things_dict['unbiased_testset_pru']
        train_loader_pru = DataLoader(trainset_pru, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=dataset_ori.train_collate)
        val_loader_pru = DataLoader(valset_pru, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=dataset_ori.collate)
        biased_test_loader_pru = DataLoader(biased_testset_pru, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=dataset_ori.collate)
        unbiased_test_loader_pru = DataLoader(unbiased_testset_pru, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=dataset_ori.collate)

        rewind_weight_c = things_dict['rewind_weight_c']
        rewind_weight_b = things_dict['rewind_weight_b']
        rewind_weight2 = things_dict['rewind_weight2']
        model_mask_dict_c = things_dict['model_mask_dict_c']
        model_mask_dict_b = things_dict['model_mask_dict_b']
        model_c.load_state_dict(rewind_weight_c)

        model_b.load_state_dict(rewind_weight_b)
        pruning.pruning_model_by_mask(model_c, model_mask_dict_c)

        pruning.pruning_model_by_mask(model_b, model_mask_dict_b)

        masker_c.load_state_dict(rewind_weight2)
        
    else:
        trainset_pru = copy.deepcopy(trainset_ori)
        valset_pru = copy.deepcopy(valset_ori)
        biased_testset_pru = copy.deepcopy(biased_testset_ori)
        unbiased_testset_pru = copy.deepcopy(unbiased_testset_ori)

        train_loader_pru = DataLoader(trainset_pru, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=dataset_ori.train_collate)
        val_loader_pru = DataLoader(valset_pru, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=dataset_ori.collate)
        biased_test_loader_pru = DataLoader(biased_testset_pru, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=dataset_ori.collate)
        unbiased_test_loader_pru = DataLoader(unbiased_testset_pru, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=dataset_ori.collate)

        #rewind_weight_c = copy.deepcopy(model_c.state_dict())

        #rewind_weight_b = copy.deepcopy(model_b.state_dict())
        #rewind_weight2 = copy.deepcopy(masker_c.state_dict())    
    
    optimizer = optim.Adam([{'params': model.parameters(), 'lr': 0.01 }], weight_decay=0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.5, patience=20, verbose=True)
    run_time, best_val_acc, best_epoch, update_biased_test_acc, update_unbiased_test_acc  = 0, 0, 0, 0, 0

    best_unbiased_test_acc, best_biased_test_acc = 0, 0
    print("done! cost time:[{:.2f} min]".format((time.time() - t0) / 60))
    for epoch in range(args.mask_epochs):

        t0 = time.time()
        epoch_train_loss, epoch_train_acc, optimizer = train_model_and_masker(model, optimizer, device, train_loader_pru, epoch, args)
        epoch_time = time.time() - t0
        run_time += epoch_time

        epoch_val_loss, epoch_val_acc = eval_acc_with_mask(model, device, val_loader_pru, epoch, val = True)
        _, epoch_biased_test_acc = eval_acc_with_mask(model, device, biased_test_loader_pru, epoch)     
        _, epoch_unbiased_test_acc = eval_acc_with_mask(model,  device, unbiased_test_loader_pru, epoch)     

        scheduler.step(epoch_val_loss)
        epoch_time = time.time() - t0
        run_time += epoch_time

        if epoch_val_acc > best_val_acc:

            best_val_acc = epoch_val_acc
            update_biased_test_acc = epoch_biased_test_acc
            update_unbiased_test_acc = epoch_unbiased_test_acc
            best_epoch = epoch
            #best_masker_state_dict = copy.deepcopy(masker_c.state_dict())

        if epoch_biased_test_acc > best_biased_test_acc:
            best_biased_test_acc = epoch_biased_test_acc
        if epoch_unbiased_test_acc > best_unbiased_test_acc:
            best_unbiased_test_acc = epoch_unbiased_test_acc

        print('-'*120)
        str1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' | ' + 'Train IMP:[{}] | Epoch [{}/{}] Loss:[{:.4f}] Train:[{:.2f}] Val:[{:.2f}] | Biased Test:[{:.2f}] Update Biased Test:[{:.2f}] Best Biased Test:[{:.2f}] | Unbiased Test:[{:.2f}] Update Unbiased Test:[{:.2f}] Best Unbiased Test:[{:.2f}] epoch:[{}] | Time:[{:.2f} min] '.format(imp_num,
                        epoch + 1, 
                        args.mask_epochs,
                        epoch_train_loss, 
                        epoch_train_acc * 100,
                        epoch_val_acc * 100, 
                        epoch_biased_test_acc * 100, 
                        update_biased_test_acc * 100,
                        best_biased_test_acc * 100,
                        epoch_unbiased_test_acc * 100, 
                        update_unbiased_test_acc * 100,
                        best_unbiased_test_acc * 100,
                        best_epoch,
                        run_time / 60,
                    ) + '\n'
        with open(filename, 'a') as result_file:
            result_file.write(str1)
        result_file.close()

        print(str1) 
    with open(filename, 'a') as result_file:
        result_file.write(str(epoch_biased_test_acc * 100)+'\n')
    result_file.close()

    things_dict = {}

    return things_dict

    
def main():  

    args = pruning.parser_loader().parse_args()
    pruning.setup_seed(args.seed)
    pruning.print_args(args)  
   
    with open(args.config) as f:
        config = json.load(f)
    
    DATASET_NAME = config['dataset']
    print("DATASET_NAME", DATASET_NAME)
    dataset = LoadData(DATASET_NAME, args.dataset) #args.dataset = MNIST_75sp_0.8
    params = config['params']
    params['seed'] = int(args.seed)
    net_params = config['net_params']
    net_params['model'] = config['model'] 
    net_params['batch_size'] = params['batch_size']
    net_params['in_dim'] = dataset.train[0][0].ndata['feat'][0].size(0)

    net_params['num_nodes'] = 75
    net_params['num_fea'] = 5
    net_params['temperature'] = 1
    
    net_params['in_dim_edge'] = 0
    num_classes = len(np.unique(np.array(dataset.train[:][1])))
    net_params['n_classes'] = num_classes

    things_dict = None
    filename = './results/'+args.out_dir +' '+str(params['seed'])+ '.txt'
    if os.path.exists(filename):
        os.remove(filename)  

    for imp_num in range(1, 2):

        things_dict = train_get_mask(dataset, net_params, things_dict, imp_num, filename, args)

    
if __name__ == '__main__':
    main()


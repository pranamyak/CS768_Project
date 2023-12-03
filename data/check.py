import os
import pickle
import dgl
import torch
import networkx as nx

print("LOADING")
path = '/mnt/nas/pkulkarni/debiasGNN_DisC/Disc_source_code/data/MNIST_75sp_0.9.pkl'
with open(path,"rb") as f:
    f = pickle.load(f)
    train = f[0]
    print("type train: ", type(train))
    val = f[1]
    print("type val: ", type(val))
    biased_test = f[2]
    print("biased test: ", type(biased_test))
    unbiased_test = f[3]
    print("unbiased test: ", type(unbiased_test))
print("DONE LOADING")



# Convert DGL graphs to NetworkX
nx_data = [dgl.to_networkx(dataset[0]) for dataset in f]

# Save NetworkX graphs
save_path = '/mnt/nas/pkulkarni/debiasGNN_DisC/Disc_source_code/data/MNIST_75sp_0.9_nx.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(nx_data, f)

# dgl.save_graphs(save_path, [train, val, biased_test, unbiased_test])
# print("DONE SAVING")
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    MLP Layer used after graph vector representation
"""
'''
class MLPReadout(nn.Module):

    def __init__(self, units, L=2): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = []
        list_FC_layers.append(nn.Linear(units[0], units[1], bias=True))

        list_FC_layers.append(nn.Linear(units[1], units[2], bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        y1 = self.FC_layers[0](y)
        y2 = nn.ReLU(inplace=True)(y1)
        y3 = self.FC_layers[1](y2)
        y4 = nn.ReLU(inplace=True)(y3)
        return y4
'''
class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y

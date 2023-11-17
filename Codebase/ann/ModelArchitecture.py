import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from torch.utils.data import TensorDataset

class Net(torch.nn.Module):
    def __init__(self, in_features, hidden_nodes_1, hidden_nodes_2,hidden_nodes_3, drop_out=0.6):
        super(Net,self).__init__()
        self.drop_out = drop_out
        in_features = in_features
        out = hidden_nodes_1
        out_2 = hidden_nodes_2
        out_3 = hidden_nodes_3
        self.ln1 = Linear(in_features, out)
        self.ln2 = Linear(out, out_2)
        self.ln3 = Linear(out_2,out_3)
        self.ln4 = Linear(out_3,1)

    def forward(self,x):
        rate = self.drop_out 
        x = torch.relu(self.ln1(x))
        x = F.dropout(x, p = rate, training = self.training)
        x = torch.relu(self.ln2(x))
        x = F.dropout(x, p = rate, training = self.training)
        x = torch.relu(self.ln3(x))
        x = F.dropout(x, p = rate, training = self.training)
        x = torch.sigmoid(self.ln4(x))
        return x
    
    def compute_l1_loss(self, w):
        return torch.abs(w).sum()

    def compute_l2_loss(self, w):
        return torch.square(w).sum()
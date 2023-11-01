import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class Linearnet(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(Linearnet, self).__init__()

        self.L1 = nn.Linear(n_inputs, n_outputs)
        init.xavier_uniform_(self.L1.weight)


    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = self.L1(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

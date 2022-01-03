from FCNN import FCNN
import torch

import torch.nn.functional as F
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, block_size, use_batch_norm):
        super(ResBlock, self).__init__()
        self.bn1 = None
        self.layer_1 = nn.Linear(block_size, block_size)
        self.layer_2 = nn.Linear(block_size, block_size)
        self.relu = nn.ReLU(inplace = True)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(block_size)
            self.bn2 = nn.BatchNorm1d(block_size)

    def forward(self, x):
        residual = x
        if self.bn1 is None:
            x = self.relu(self.layer_1(x))
            x = self.relu(self.layer_2(x) + residual)
        else:
            x = self.relu(self.bn1(self.layer_1(x)))
            x = self.relu(self.bn2(self.layer_2(x)) + residual)
        return x

class ResNN(FCNN):
    def __init__(self, layers, num_res_blocks = 4, use_batch_norm = True):
        super(ResNN, self).__init__(layers)
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            self.res_blocks.append(ResBlock(layers[-2], use_batch_norm = use_batch_norm))
        self.params.append(self.res_blocks)

    def forward(self, x):
        for i in range(len(self.fc) - 1):
            x = F.relu(self.fc[i](x))
        for i in range(len(self.res_blocks)):
            x = self.res_blocks[i](x)
        return self.fc[-1](x)

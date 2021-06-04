from FCNN import FCNN
import torch

import torch.nn.functional as F
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, block_size):
        super(ResBlock, self).__init__()
        self.layer_1 = nn.Linear(block_size, block_size)
        self.layer_2 = nn.Linear(block_size, block_size)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        residual = x
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x) + residual)
        return x

class ResNN(FCNN):
    def __init__(self, layers, num_res_blocks = 4):
        super(ResNN, self).__init__(layers)
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            self.res_blocks.append(ResBlock(layers[-2]))
        self.params.append(self.res_blocks)

    def forward(self, x):
        for i in range(len(self.fc) - 1):
            x = F.relu(self.fc[i](x))
        for i in range(len(self.res_blocks)):
            x = self.res_blocks[i](x)
        return self.fc[-1](x)

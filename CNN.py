from FCNN import FCNN
import torch

import torch.nn.functional as F
import torch.nn as nn

class CNN(FCNN):
    def __init__(self, layers, num_cnn_layers = 2, input_channels = 9):
        super(CNN, self).__init__(layers)

        self.convs = nn.ModuleList()

        dim = input_channels
        filters = 32
        for i in range(num_cnn_layers):
            self.convs.append(nn.Conv2d(dim, filters, kernel_size = 3, padding = 1))
            dim = filters
        layers[0] = 22 * 22 * filters
        self.dim = layers[0]
        self.fc = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.fc.append(nn.Linear(layers[i], layers[i+1]))

        self.params = nn.ModuleList()
        self.params.append(self.fc)
        self.params.append(self.convs)

    def forward(self, x):

        for i in range(len(self.convs)):
            x = F.relu(self.convs[i](x))
        x = x.view(-1, self.dim)
        for i in range(len(self.fc) - 1):
            x = F.relu(self.fc[i](x))
        return self.fc[-1](x)

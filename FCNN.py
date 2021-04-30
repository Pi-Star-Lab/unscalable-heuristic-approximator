import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from collections import defaultdict

class search_expanded_loss(nn.Module):
    def __init__(self, branching_factor = 3):
        super(search_expanded_loss, self).__init__()
        self.bf = branching_factor

    def forward(self, target, output):
        distance_from_goal = target[:, 0]
        weight = 1 / self.bf ** (distance_from_goal)
        return torch.mean(weight * (target[:, 0] - output[:, 0]) ** 2)

class balanced_data_loss(nn.Module):
    def __init__(self):
        super(balanced_data_loss, self).__init__()

    def get_weights(self, values):
        rounded_values = torch.round(values[:, 0])
        _, inverses, counts = torch.unique(rounded_values, \
                return_counts = True, return_inverse = True)
        return counts[inverses[range(rounded_values.shape[0])]]

    def forward(self, target, output):
        w = self.get_weights(target)
        max_w = torch.max(w)
        #print(max_w.item() / w.float())
        #print(max_w.item() / w)
        return torch.mean((max_w.item() / w.float()) * ((target[:, 0] - output[:, 0]) ** 2))

class FCNN(nn.Module):
    def __init__(self, layers):
        super(FCNN, self).__init__()
        self.device = None
        self.fc = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.fc.append(nn.Linear(layers[i], layers[i+1]))

    def forward(self, x):
        for i in range(len(self.fc) - 1):
            x = F.relu(self.fc[i](x))
        x = self.fc[-1](x)
        return x

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))

    def compile(self, loss_fn = nn.MSELoss, optimizer = optim.Adam, lr=1e-3):

        self.loss_fn = loss_fn()
        self.optimizer = optimizer(self.parameters(), lr = lr)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_fn.to(self.device)
        self.to(self.device)

    def predict(self, x, batch_size = 1e3):
        batch_size = int(min(x.shape[0], batch_size))
        n_batches = math.ceil(x.shape[0] / batch_size)
        y = []
        with torch.no_grad():
            self.eval()
            x = torch.Tensor(x)
            for i in range(n_batches):
                local_x = x[i * batch_size:(i+1)*batch_size,].to(self.device)
                y.append(self.forward(local_x))
        return torch.cat(y)

    def run_epoch(self, x, y, batch_size = 1e5, verbose = 1):

        if self.device is None:
            Exception("Make sure you compile the model first!")

        x = torch.Tensor(x)
        y = torch.unsqueeze(torch.Tensor(y), 1)
        self.train()
        batch_size = min(x.shape[0], batch_size)
        n_batches = math.ceil(x.shape[0] / batch_size)
        running_loss = 0.
        for i in range(n_batches):
            local_x, local_y = x[i*batch_size:(i+1)*batch_size,], \
                    y[i*batch_size:(i+1)*batch_size,]

            local_x, local_y = local_x.to(self.device), local_y.to(self.device)

            self.optimizer.zero_grad()
            pred = self.forward(local_x)
            loss = self.loss_fn(local_y, pred)
            loss.backward()

            self.optimizer.step()
            running_loss += loss.item()

        if verbose == 1:
            print("Samples used: {} Epoch Loss:{}".format(x.shape[0], running_loss))

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_weights(self):
        return self.state_dict()

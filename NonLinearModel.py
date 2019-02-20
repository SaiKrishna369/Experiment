import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, input_size, output_size):
        super(Network, self).__init__()
        self.hidden1 = nn.Linear(input_size, 64)
        self.hidden2 = nn.Linear(64, 128)
        self.output = nn.Linear(128, output_size)
    
    def forward(self, x):
        h1 = torch.sigmoid(self.hidden1(x))
        h2 = torch.tanh(self.hidden2(h1))
        out = self.output(h2)
        return out
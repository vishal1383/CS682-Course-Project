from torch import nn
import torch

class MLP(nn.Module):
    def __init__(self, input_dim = 512, output_dim = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 250)
        self.fc2 = nn.Linear(250, 100)
        self.fc3 = nn.Linear(100, output_dim)

        self.relu = nn.ReLU()
        return

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
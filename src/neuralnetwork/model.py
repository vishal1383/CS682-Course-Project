from torch import nn
import torch.nn.init as init

class MLP(nn.Module):
    def __init__(self, input_dim = 512, output_dim = 2):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 512)
        self.dropout1 = nn.Dropout()
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout()
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, output_dim)

        self.relu = nn.ReLU()
        
        self._initialize_weights()
        return
    
    def _initialize_weights(self):
        init.kaiming_uniform_(self.fc1.weight, nonlinearity = 'relu')
        init.kaiming_uniform_(self.fc2.weight, nonlinearity = 'relu')
        init.kaiming_uniform_(self.fc3.weight, nonlinearity = 'relu')
        init.xavier_uniform_(self.fc4.weight)
        
        init.zeros_(self.fc1.bias)
        init.zeros_(self.fc2.bias)
        init.zeros_(self.fc3.bias)
        init.zeros_(self.fc4.bias)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set device to CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the 2-layer MLP
class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayerMLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

# Example dimensions
input_dim = 100   # adjust to your input features
hidden_dim = 64
output_dim = 10   # number of classes

# Instantiate and move model to GPU
model = TwoLayerMLP(input_dim, hidden_dim, output_dim).to(device)

# Example input: random tensor
x = torch.randn(32, input_dim).to(device)  # batch of 32

# Forward pass on CUDA
output = model(x)
print(output)

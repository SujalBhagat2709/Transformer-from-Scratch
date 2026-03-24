import torch.nn as nn

class MultiHeadAttention(nn.Module):
    
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, x):
        return self.q(x), self.k(x), self.v(x)
import torch.nn as nn

class TransformerBlock(nn.Module):
    
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=2)
        self.ff = nn.Linear(dim, dim)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        return self.ff(attn_output)
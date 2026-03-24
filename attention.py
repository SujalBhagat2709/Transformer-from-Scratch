import torch

def attention(q, k, v):
    
    scores = torch.matmul(q, k.transpose(-2, -1))
    
    weights = torch.softmax(scores, dim=-1)
    
    return torch.matmul(weights, v)
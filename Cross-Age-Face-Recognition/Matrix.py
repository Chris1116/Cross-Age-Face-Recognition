from numpy import dot
from numpy.linalg import norm
import torch
import torch.nn as nn

def get_cosine_similarity(feat1, feat2):
    
    cos_sim = dot(feat1, feat2)/(norm(feat1)*norm(feat2))
    return cos_sim


class Final(nn.Module):
    def __init__(self):
        super().__init__()
        self.cs = nn.CosineSimilarity()
        self.lin = nn.Linear(1, 1)

    def forward(self, x, y):
        t = self.cs(x, y)
        return self.lin(t.reshape(-1, 1))


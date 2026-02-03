import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    def __init__(self, dim, device):
        super().__init__()
        self.dim = dim
        self.device = device
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))

    def forward(self, t):
        t = t.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        return torch.cat((t.sin(), t.cos()), dim=-1)

class SimpleDenoiser(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, device='cpu'):
        super().__init__()
        self.device = device
        self.time_embed = TimeEmbedding(hidden_dim, device)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.to(device)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        x_emb = self.input_proj(x)
        h = x_emb + t_emb
        return self.net(h)
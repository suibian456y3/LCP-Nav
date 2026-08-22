import torch
import torch.nn as nn

class RobustDynamics(nn.Module):
    def __init__(self, encoding_size, action_embed_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(encoding_size + action_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, encoding_size),
        )
        self.out_norm = nn.LayerNorm(encoding_size)

    def forward(self, z_t, a_emb):
        return self.out_norm(z_t + self.net(torch.cat([z_t, a_emb], dim=-1)))
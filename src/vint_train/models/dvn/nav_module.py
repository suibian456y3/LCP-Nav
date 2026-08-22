import torch.nn as nn

class TokenPooler(nn.Module):
    def __init__(self, in_dim=1024, out_dim=768, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, tokens):
        x = tokens.mean(dim=2)
        x = self.proj(x)
        return x
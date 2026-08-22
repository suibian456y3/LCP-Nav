import torch
import torch.nn as nn

class BottleneckAdapter(nn.Module):
    def __init__(self, in_features, bottleneck_dim=64, dropout=0.0, init_scale=1e-3):

        super().__init__()
        self.fuse = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, in_features // 2),
        )
        self.adapter = nn.Sequential(
            nn.LayerNorm(in_features // 2),
            nn.Linear(in_features // 2, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, in_features // 2)
        )
        
        # Zero Initialization
        # nn.init.zeros_(self.adapter[-1].weight)
        # nn.init.zeros_(self.adapter[-1].bias)
        # self.scale = nn.Parameter(torch.tensor(init_scale))

    def forward(self, x):
        x = self.fuse(x)
        return x + self.adapter(x)
        # return x + self.scale * self.adapter(x)

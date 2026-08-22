import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class DepthMultiQueryPooling(nn.Module):
    """
    Multi-query attention pooling over depth tokens.

    Input:
        x: [B, N, D] depth tokens (flattened spatial grid)
    Output:
        out: [B, Q, D] pooled tokens (Q=4 by default)
    Optional:
        grid_hw: (H, W) depth token grid size. If provided, we can add soft spatial priors.
    """
    def __init__(
        self,
        embed_dim: int,
        num_queries: int = 4,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_spatial_priors: bool = True,
        sigma: float = 0.35,   # controls how wide the soft prior is
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.use_spatial_priors = use_spatial_priors
        self.sigma = sigma

        self.queries = nn.Parameter(torch.empty(1, num_queries, embed_dim))
        nn.init.normal_(self.queries, std=0.02)

        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

        # query residual mixing (prevents queries dominating early)
        self.alpha = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2)≈0.12 small residual initially

    @staticmethod
    def _make_grid(H: int, W: int, device) -> torch.Tensor:
        # returns [N, 2] with coords in [-1,1], order row-major
        ys = torch.linspace(-1.0, 1.0, steps=H, device=device)
        xs = torch.linspace(-1.0, 1.0, steps=W, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # [N,2] (x,y)
        return grid

    def _spatial_bias(self, grid_hw: Tuple[int, int], device) -> torch.Tensor:
        """
        Create additive attention bias of shape [Q, N] to encourage specialization:
        queries: [front, left, right, center] by default ordering.
        """
        H, W = grid_hw
        grid = self._make_grid(H, W, device=device)  # [N,2]
        x = grid[:, 0]  # [-1,1], left=-1, right=+1
        y = grid[:, 1]  # [-1,1], top=-1, bottom=+1 (depends on flatten order; it's just a soft prior)

        # Define 4 Gaussian centers in (x,y)
        # You can tweak these centers depending on your camera/view convention:
        centers = torch.tensor([
            [0.0, -0.7],   # front: towards top
            [-0.7, 0.0],   # left
            [0.7, 0.0],    # right
            [0.0, 0.0],    # center
        ], device=device)  # [4,2]

        if self.num_queries != 4:
            # If you change num_queries, either provide your own centers or disable priors.
            # Here we fallback to zeros.
            return torch.zeros(self.num_queries, H * W, device=device)

        # Gaussian log-prior (additive bias to attention logits)
        # bias[q, n] = -||grid[n] - centers[q]||^2 / (2*sigma^2)
        diff = grid.unsqueeze(0) - centers.unsqueeze(1)  # [4,N,2]
        dist2 = (diff ** 2).sum(dim=-1)                  # [4,N]
        bias = -dist2 / (2 * (self.sigma ** 2))          # [4,N]

        return bias  # [4,N], larger = more preferred

    def forward(self, x: torch.Tensor, grid_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        x: [B, N, D]
        return: [B, Q, D]
        """
        B, N, D = x.shape
        assert D == self.embed_dim, f"embed_dim mismatch: x has {D}, module expects {self.embed_dim}"

        q = self.queries.expand(B, -1, -1)  # [B,Q,D]

        attn_mask = None
        if self.use_spatial_priors and grid_hw is not None:
            H, W = grid_hw
            assert H * W == N, f"grid_hw {grid_hw} implies N={H*W}, but got N={N}"
            bias = self._spatial_bias(grid_hw, device=x.device)  # [Q,N]
            # PyTorch MHA: attn_mask shape can be [Q, N] and is added to attention logits.
            attn_mask = bias

        attn_out, _ = self.mha(query=q, key=x, value=x, attn_mask=attn_mask, need_weights=False)  # [B,Q,D]

        out = self.norm(attn_out + torch.sigmoid(self.alpha) * q)
        return out

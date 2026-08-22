import torch
from torch import nn


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian regularizer for latent embeddings.

    The input is expected to be a sequence of embeddings with shape
    (B, T, D), where the statistic is estimated over the batch dimension.
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj

        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(1)
        if embeddings.dim() != 3:
            raise ValueError(f"SIGReg expects (B, T, D) or (B, D), got {embeddings.shape}")

        projections = torch.randn(
            embeddings.size(-1),
            self.num_proj,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        projections = projections / projections.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12)

        projected = (embeddings @ projections).unsqueeze(-1) * self.t.to(embeddings.dtype)
        err = (
            (projected.cos().mean(dim=0) - self.phi.to(embeddings.dtype)).square()
            + projected.sin().mean(dim=0).square()
        )
        statistic = (err @ self.weights.to(embeddings.dtype)) * embeddings.size(0)
        return statistic.mean()

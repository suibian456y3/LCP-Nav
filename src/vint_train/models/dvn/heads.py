import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHypothesisHead(nn.Module):
    def __init__(self, input_dim, num_hypotheses=5, len_traj_pred=5, action_dim=4):
        super().__init__()
        self.K = num_hypotheses
        self.T = len_traj_pred
        self.A = action_dim

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.hyp_emb = nn.Embedding(self.K, 32)

        self.traj_head = nn.Sequential(
            nn.Linear(128 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, self.T * self.A),
        )

    def forward(self, x):
        B = x.shape[0]
        base = self.backbone(x)  # [B,128]

        hyp_ids = torch.arange(self.K, device=x.device)
        h_emb = self.hyp_emb(hyp_ids)  # [K,32]

        base_rep = base.unsqueeze(1).expand(-1, self.K, -1)
        h_emb = h_emb.unsqueeze(0).expand(B, -1, -1)

        feat = torch.cat([base_rep, h_emb], dim=-1)  # [B,K,160]
        raw = self.traj_head(feat).view(B, self.K, self.T, self.A)

        dx = torch.tanh(raw[..., 0:1]) * 1.5
        dy = torch.tanh(raw[..., 1:2]) * 1.5

        ang = F.normalize(raw[..., 2:], dim=-1)

        return torch.cat([dx, dy, ang], dim=-1)


class CandidateConditionedScorer(nn.Module):
    """
    对每条候选轨迹单独评分
    输入:
      - z_obs
      - z_goal
      - z_end (rollout终点latent)
      - traj
    """
    def __init__(
        self,
        state_dim: int,
        traj_len: int,
        action_dim: int,
        traj_embed_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.traj_len = traj_len
        self.action_dim = action_dim

        self.traj_encoder = nn.Sequential(
            nn.Linear(traj_len * action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, traj_embed_dim),
            nn.ReLU(),
        )

        in_dim = state_dim * 4 + traj_embed_dim
        self.score_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, z_obs, z_goal, z_end, trajs):
        """
        Args:
            z_obs: [B, D]
            z_goal: [B, D]
            z_end: [B, K, D]
            trajs: [B, K, T, A]
        Returns:
            scores: [B, K]
        """
        B, K, T, A = trajs.shape
        assert T == self.traj_len
        assert A == self.action_dim

        traj_feat = self.traj_encoder(trajs.reshape(B * K, T * A)).view(B, K, -1)

        z_obs_rep = z_obs.unsqueeze(1).expand(-1, K, -1)
        z_goal_rep = z_goal.unsqueeze(1).expand(-1, K, -1)

        score_in = torch.cat(
            [
                z_obs_rep,
                z_goal_rep,
                z_end,
                z_goal_rep - z_end,
                traj_feat,
            ],
            dim=-1,
        )

        scores = self.score_mlp(score_in).squeeze(-1)  # [B, K]
        return scores
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHypothesisHead(nn.Module):
    def __init__(self, input_dim, num_hypotheses=5, len_traj_pred=5, action_dim=2):
        """
        Multi-Hypothesis Prediction Head.
        Outputs K independent trajectories and K probability scores.
        """
        super().__init__()
        self.K = num_hypotheses
        self.T = len_traj_pred
        self.A = action_dim
        
        # Shared intermediate layer
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # 1. Trajectory Prediction Head: Outputs K * T * A
        # Reshaped to [B, K, T, A]
        self.traj_head = nn.Linear(128, self.K * self.T * self.A)
        
        # 2. Scoring Head: Outputs K scores (logits)
        # Reshaped to [B, K]
        self.score_head = nn.Linear(128, self.K)
        
    def forward(self, x):
        """
        Args:
            x: [B, input_dim] feature vector
        Returns:
            trajs: [B, K, T, A]
            scores: [B, K] (logits)
        """
        B = x.shape[0]
        feat = self.proj(x)
        
        # Predict K trajectories
        trajs_flat = self.traj_head(feat)
        trajs = trajs_flat.view(B, self.K, self.T, self.A)
        
        # Predict K scores
        scores = self.score_head(feat)
        
        return trajs, scores

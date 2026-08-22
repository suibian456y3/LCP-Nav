import torch
import torch.nn as nn
from .safety_utils import project_and_sample

class DifferentiableCollisionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, trajs, depth_img, intrinsics, camera_height=1.0, safety_radius=0.4):
        """
        Args:
            trajs: [B, K, T, 2]
            depth_img: [B, 1, H, W]
            intrinsics: [B, 3, 3]
        """
        if len(trajs.shape) == 3: trajs = trajs.unsqueeze(1)
        
        cost_per_hypothesis = project_and_sample(trajs, depth_img, intrinsics, camera_height, safety_radius)
        loss = cost_per_hypothesis.mean()
        return loss

    def get_default_intrinsics(self, B, device):
        K = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
        K[:, 0, 0] = 150.0 
        K[:, 1, 1] = 150.0 
        K[:, 0, 2] = 112.0 
        K[:, 1, 2] = 112.0 
        return K

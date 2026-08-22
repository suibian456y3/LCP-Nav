import torch
import numpy as np
import torch.nn as nn
from .safety_utils import project_and_sample

class DifferentiableCollisionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, trajs, depth_img, intrinsics=None, camera_height=None, safety_radius=0.4, car='agilex'):
        """
        Args:
            trajs: [B, K, T, 2]
            depth_img: [B, 1, H, W]
            intrinsics: [B, 3, 3]
        """
        if len(trajs.shape) == 3: trajs = trajs.unsqueeze(1)
        B = trajs.shape[0]
        if intrinsics is None or camera_height is None:
            intrinsics_default, camera_height_default = self.get_default_camera_config(
                B, trajs.device, car=car
            )
            if intrinsics is None:
                intrinsics = intrinsics_default
            if camera_height is None:
                camera_height = camera_height_default

        cost_per_hypothesis = project_and_sample(
            trajs, depth_img, intrinsics, camera_height, safety_radius
        )
        return cost_per_hypothesis

    def get_default_camera_config(self, B, device, car='drivebot'):
        intrinsics_np = np.load('data/intrinsics.npy', allow_pickle=True)
        intrinsics_np = intrinsics_np.item()
        intrinsics = intrinsics_np[car]['intrinsics']
        camera_height = intrinsics_np[car]['camera_height']
        return torch.from_numpy(intrinsics).unsqueeze(0).repeat(B, 1, 1).to(device), camera_height

import torch
import torch.nn.functional as F

def project_and_sample(trajs, depth_img, intrinsics, camera_height=1.0, safety_radius=0.4):
    """
    Project 3D waypoints to 2D depth image and sample depth values.
    Returns the 'penalty' (ReLU of mismatch) for each point.
    
    Args:
        trajs: [B, K, T, 2]
        depth_img: [B, 1, H, W]
        intrinsics: [B, 3, 3]
    Returns:
        collision_cost: [B, K] Average collision cost per hypothesis
    """
    B, K, T, _ = trajs.shape
    _, _, H, W = depth_img.shape
        
    x_robot = trajs[..., 0] 
    y_robot = trajs[..., 1]
    
    X_cam = -y_robot        
    Y_cam = torch.full_like(x_robot, camera_height)
    Z_cam = x_robot         
    Z_cam = torch.clamp(Z_cam, min=0.1)
    
    fx = intrinsics[:, 0, 0].view(B, 1, 1)
    fy = intrinsics[:, 1, 1].view(B, 1, 1)
    cx = intrinsics[:, 0, 2].view(B, 1, 1)
    cy = intrinsics[:, 1, 2].view(B, 1, 1)
    
    u = (fx * X_cam / Z_cam) + cx
    v = (fy * Y_cam / Z_cam) + cy
    
    u_norm = 2 * u / (W - 1) - 1
    v_norm = 2 * v / (H - 1) - 1
    
    grid = torch.stack([u_norm, v_norm], dim=-1) # [B, K, T, 2]
    
    # Sample Depth: [B, 1, K, T]
    # Note: grid_sample interprets [B, H_grid, W_grid, 2] as spatial dims
    # We treat K as H_grid, T as W_grid
    observed_depth = F.grid_sample(depth_img, grid, align_corners=False, padding_mode='border')
    observed_depth = observed_depth.squeeze(1) # [B, K, T]
    
    # Calculate Cost
    safe_dist = Z_cam
    penalty = F.relu((safe_dist - safety_radius) - observed_depth)
    
    # Valid mask
    valid_mask = (u_norm > -1) & (u_norm < 1) & (v_norm > -1) & (v_norm < 1)
    penalty = penalty * valid_mask.float()
    
    # Sum over T to get cost per hypothesis
    cost_per_hypothesis = penalty.mean(dim=2) # [B, K]
    
    return cost_per_hypothesis

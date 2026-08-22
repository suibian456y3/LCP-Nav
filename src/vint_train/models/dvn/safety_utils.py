import torch
import torch.nn.functional as F

def project_and_sample(trajs, depth_img, intrinsics, camera_height, safety_radius=0.4, robot_width=0.6, max_detect_dist=3.5):
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
    half_w = robot_width / 2.0
    
    # 1. 定义采样偏移量: 中心、左边缘、右边缘
    offsets = torch.tensor([0.0, -half_w, half_w], device=trajs.device)
    # 扩展轨迹维度并加入偏移量
    expanded_trajs = trajs.unsqueeze(3).repeat(1, 1, 1, 3, 1)
    expanded_trajs[..., 1] += offsets.view(1, 1, 1, 3)
    # 展平以便统一投影计算
    flat_trajs = expanded_trajs.view(B, -1, 2)

    # 坐标转换，body系转相机系
    x_robot = flat_trajs[..., 0] 
    y_robot = flat_trajs[..., 1]
    
    X_cam = -y_robot        
    Y_cam = torch.full_like(x_robot, camera_height)
    Z_cam = torch.clamp(x_robot - 0.20, min=0.1)
    
    # 2. 投影到像素坐标系(u, v)
    fx = intrinsics[:, 0, 0].view(B, 1)
    fy = intrinsics[:, 1, 1].view(B, 1)
    cx = intrinsics[:, 0, 2].view(B, 1)
    cy = intrinsics[:, 1, 2].view(B, 1)
    
    u = (fx * X_cam / Z_cam) + cx
    v = (fy * Y_cam / Z_cam) + cy
    
    u_norm = 2 * u / (W - 1) - 1
    v_norm = 2 * v / (H - 1) - 1
    grid = torch.stack([u_norm, v_norm], dim=-1).view(B, K, T * 3, 2)
    
    # 3. 采样深度图，获取轨迹点的观测深度
    observed_depth = F.grid_sample(depth_img, grid, align_corners=False, padding_mode='border')
    observed_depth = observed_depth.squeeze(1).view(B, K, T, 3)
    
    # 4. 计算碰撞惩罚，(Z_cam-半径) > 观测深度，代表碰撞到物体，计算一个碰撞损失，否则为0(ReLU)
    z_vals = Z_cam.view(B, K, T, 3)
    penalty = F.relu((z_vals - safety_radius) - observed_depth)
    
    # 深度裁剪, 忽略过远的障碍信息
    near_mask = (observed_depth < max_detect_dist)
    penalty = penalty * near_mask.float()

    dist_weight = torch.exp(-z_vals / 2.0)
    penalty = penalty * dist_weight

    # 视野边界处理, 如果点跑到了视野外 (u_norm > 1 或 < -1)，我们不能假设它是安全的, 因此给一个较小的固定惩罚0.05
    valid_mask = (u_norm > -1.0) & (u_norm < 1.0) & (v_norm > -1.0) & (v_norm < 1.0)
    valid_mask = valid_mask.view(B, K, T, 3)
    penalty = torch.where(valid_mask, penalty, torch.tensor(0.1, device=trajs.device))

    penalty_max, _ = torch.max(penalty, dim=3)
        
    return penalty_max.mean(dim=2)

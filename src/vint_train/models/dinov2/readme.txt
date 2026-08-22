2026.1.12修改：
1. 重写model的train方法，让model.train()时，dinov2仍然是eval()
2. obs和goal修改为共享Adapter
3. 修改从dinov2中拿到的latent，从x_norm_clstoken变为x_norm_patchtokens，提高他的几何特征理解能力而非语义特征理解能力，后续可：
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeometricDistanceHead(nn.Module):
    """
    专为距离估计设计的 Head，通过对比当前特征和目标特征的差异来预测距离。
    """
    def __init__(self, feature_dim, hidden_dim=256):
        super().__init__()
        # 输入维度是 feature_dim * 4 (curr, goal, diff, product)
        input_dim = feature_dim * 4
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.ReLU() # 距离永远为正
        )

    def forward(self, z_curr, z_goal):
        # z_curr, z_goal: [B, D]
        diff = z_curr - z_goal
        abs_diff = torch.abs(diff)
        product = z_curr * z_goal
        
        # 显式拼接特征差异信息
        # 这种方式能极大地帮助模型感知两张图在特征空间中的“位移”
        combined = torch.cat([z_curr, z_goal, abs_diff, product], dim=-1)
        return self.net(combined)

class SpatialPatchPooler(nn.Module):
    """
    将 DINOv2 的 Patch Tokens 聚合为具有几何感知能力的特征向量
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.proj = nn.Linear(128, out_channels)

    def forward(self, patch_tokens):
        # patch_tokens: [B, N, D], DINOv2 ViT-B/14 产生 16x16=256 个 patches (对于224输入)
        B, N, D = patch_tokens.shape
        H = W = int(N**0.5)
        x = patch_tokens.transpose(1, 2).reshape(B, D, H, W)
        x = self.conv(x).flatten(1) # [B, 128]
        return self.proj(x)
# 在 __init__ 中添加
self.patch_pooler = SpatialPatchPooler(obs_encoding_size, obs_encoding_size)
self.dist_predictor = GeometricDistanceHead(feature_dim=obs_encoding_size)
def forward(self, obs_img, goal_img, depth_img, ...):
        # ... 前面的 RGB 处理 ...
        with torch.no_grad():
            # 提取包含 Patch 的全量特征
            obs_feats_full = self.rgb_backbone.forward_features(obs_img_batch)
            obs_cls = obs_feats_full['x_norm_clstoken']
            obs_patch = obs_feats_full['x_norm_patchtokens']
            
            goal_feats_full = self.rgb_backbone.forward_features(goal_img)
            goal_cls = goal_feats_full['x_norm_clstoken']
            goal_patch = goal_feats_full['x_norm_patchtokens']

        # 1. 结合 CLS 和 Patch 信息 (增强几何感知)
        # 我们用 Patch Pooler 提取空间特征，并与 Adapter 后的 CLS 融合
        obs_spatial = self.patch_pooler(obs_patch)
        goal_spatial = self.patch_pooler(goal_patch)
        
        # 这里的 Adapter 依然保留，但我们让它的输出更丰富
        rgb_tokens = self.obs_adapter(obs_cls) + obs_spatial 
        goal_tokens = self.goal_adapter(goal_cls) + goal_spatial
        
        # ... 中间的 Transformer 融合逻辑保持不变 ...
        # transformer_out = self.transformer(final_seq)
        
        # 2. 修改 Distance 预测逻辑
        # 我们不再从 Transformer 的最后一位预测距离，而是直接对比“当前”和“目标”
        # 从序列中取当前帧(最后一帧)和目标帧
        z_curr = transformer_out[:, -2, :] # 当前观测在 Transformer 后的表征
        z_goal = transformer_out[:, -1, :] # 目标在 Transformer 后的表征
        
        # 使用专门的对比 Head 预测距离
        dist_pred = self.dist_predictor(z_curr, z_goal)
        
        # ... 其余 Action 预测逻辑 ...
进一步增强几何感知能力
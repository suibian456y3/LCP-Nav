import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Optional, Tuple, Dict
from efficientnet_pytorch import EfficientNet
from vint_train.models.base_model import BaseModel
from vint_train.models.dinov2.transformer_module import TransformerModule
from vint_train.models.dinov2.heads import MultiHypothesisHead
from vint_train.models.dinov2.safety_loss import DifferentiableCollisionLoss
from vint_train.models.dvn.dynamics import RobustDynamics

class DVN(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        vision_encoder: Optional[str] = "efficientformerv2_s1",
        encoding_size: Optional[int] = 512,
        late_fusion: Optional[bool] = False,
        mha_num_attention_heads: Optional[int] = 4,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
        num_hypotheses: int = 5,
    ) -> None:
        super(DVN, self).__init__(context_size, len_traj_pred, learn_angle)
        
        # 维度定义
        self.num_action_params = 4 if learn_angle else 2
        self.encoding_size = encoding_size
        self.action_embed_dim = 128
        self.policy_latent_dim = 128
        self.late_fusion = late_fusion

        # 1. 骨干网络 (Perception)
        self.obs_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=3)
        self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=3 if late_fusion else 6)
        self.num_features = self.obs_encoder._fc.in_features
        
        # 2. 特征压缩与位置嵌入
        self.compress_obs_enc = nn.Linear(self.num_features, self.encoding_size) if self.num_features != self.encoding_size else nn.Identity()
        self.compress_goal_enc = nn.Linear(self.num_features, self.encoding_size) if self.num_features != self.encoding_size else nn.Identity()
        self.type_embed = nn.Embedding(2, encoding_size)
        self.seq_len = (context_size + 1) + 1

        # 3. 时序融合 (Reasoning)
        self.transformer = TransformerModule(
            embed_dim=encoding_size, seq_len=self.seq_len,
            nhead=mha_num_attention_heads, num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor
        )

        # 4. 预测头 (Decision)
        self.policy_compressor = nn.Sequential(
            nn.LayerNorm(encoding_size),
            nn.Linear(encoding_size, self.policy_latent_dim),
            nn.GELU(),
        )
        self.action_predictor = MultiHypothesisHead(
            input_dim=self.policy_latent_dim, num_hypotheses=num_hypotheses,
            len_traj_pred=len_traj_pred, action_dim=self.num_action_params 
        )
        self.dist_predictor = nn.Sequential(
            nn.Linear(encoding_size * 4, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 1)
        )

        # 5. 世界模型 (World Model - Dynamics)
        self.action_embed = nn.Sequential(
            nn.Linear(self.num_action_params, self.action_embed_dim),
            nn.GELU(),
            nn.Linear(self.action_embed_dim, self.action_embed_dim),
        )
        self.dynamics = RobustDynamics(self.encoding_size, self.action_embed_dim, hidden_dim=512)
        self.dyn_norm = nn.LayerNorm(encoding_size)
        
        # 6. 安全与避障
        self.safety_loss_fn = DifferentiableCollisionLoss(safety_radius=0.5)

    def get_feature(
        self, obs_img: torch.Tensor, goal_img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """提取特征并返回 action_pred, action_scores, z_obs, z_goal"""
        B = obs_img.shape[0]
        obs_img_list = torch.split(obs_img, 3, dim=1)
        obs_img_batch = torch.cat(obs_img_list, dim=0)
        curr_img = obs_img_list[-1]

        # Goal Encoding
        if self.late_fusion:
            goal_feat = self.goal_encoder.extract_features(goal_img)
        else:
            goal_feat = self.goal_encoder.extract_features(torch.cat([curr_img, goal_img], dim=1))
        goal_feat = self.compress_goal_enc(self.goal_encoder._avg_pooling(goal_feat).flatten(1))
        
        # Obs Encoding
        obs_feat = self.obs_encoder.extract_features(obs_img_batch)
        obs_feat = self.compress_obs_enc(self.obs_encoder._avg_pooling(obs_feat).flatten(1))
        obs_feat = obs_feat.reshape((self.context_size+1, B, self.encoding_size)).transpose(0, 1)

        # Transformer Fusion
        tokens = torch.cat((obs_feat, goal_feat.unsqueeze(1)), dim=1)
        type_ids = torch.zeros((B, tokens.shape[1]), dtype=torch.long, device=obs_img.device)
        type_ids[:, -1] = 1 # 最后一个是 Goal
        transformer_out = self.transformer(tokens + self.type_embed(type_ids))

        z_obs = transformer_out[:, -2, :]
        z_goal = transformer_out[:, -1, :]
        
        # Predictions
        dist_pred = self.dist_predictor(torch.cat([z_obs, z_goal, z_goal-z_obs, z_goal*z_obs], dim=-1))
        policy_latent = self.policy_compressor(z_goal)
        action_pred, action_scores = self.action_predictor(policy_latent)
        
        # 轨迹积分处理 (假设输出是相对位移)
        action_pred[:, :, :2] = torch.cumsum(action_pred[:, :, :2], dim=1)
        if self.learn_angle:
            action_pred[:, :, :, 2:] = F.normalize(action_pred[:, :, :, 2:].clone(), dim=-1)

        return dist_pred, action_pred, action_scores, z_obs, z_goal

    def evaluate_imagination(self, z_t: torch.Tensor, z_goal: torch.Tensor, action_pred: torch.Tensor) -> torch.Tensor:
        """多步脑补评估 (Latent MPC)"""
        B, K, T, A = action_pred.shape
        curr_z = z_t.unsqueeze(1).repeat(1, K, 1).reshape(B * K, -1)
        z_goal_flat = z_goal.unsqueeze(1).repeat(1, K, 1).reshape(B * K, -1)
        
        for t in range(T):
            a_t = action_pred[:, :, t, :].reshape(B * K, -1)
            a_emb = self.action_embed(a_t)
            curr_z = self.dynamics(curr_z, a_emb)
            curr_z = self.dyn_norm(curr_z)
            
        return F.cosine_similarity(F.normalize(curr_z, dim=-1), F.normalize(z_goal_flat, dim=-1), dim=-1).view(B, K)

    def train_dynamics_step(self, z_sequence, action_label_sequence):
            """
            按照输入序列的实际长度进行多步预测。
            z_sequence: [B, L, D]  (L 是随机长度 2~4, 代表 1~3 步预测)
            action_label_sequence: [B, L-1, A]
            """
            B, L, D = z_sequence.shape
            steps = L - 1 # 实际能预测的步数
            
            if steps <= 0:
                return torch.tensor(0.0).to(z_sequence.device)

            # 1. 初始状态：第 0 帧作为起点
            curr_z = z_sequence[:, 0, :]
            total_loss = 0

            # 2. 按照序列长度递归预测
            for h in range(1, L):
                # 获取对应的动作 (t-1)
                a_t = action_label_sequence[:, h - 1, :]
                a_emb = self.action_embed(a_t)
                
                # 动力学预测：z_t = Dynamics(z_{t-1}, a_{t-1})
                curr_z = self.dynamics(curr_z, a_emb)
                
                # 取出真实的 GT 特征进行对比
                z_gt = z_sequence[:, h, :].detach()
                
                # 计算余弦相似度损失
                step_loss = (1 - F.cosine_similarity(curr_z, z_gt, dim=-1)).mean()
                
                # 这里的权重可以根据 h 增加（越往后权重越大，或者平均）
                total_loss += step_loss

            return total_loss / steps

    @torch.no_grad()
    def inference_with_safety(
        self, obs_img, goal_img, depth_img=None, intrinsics=None, 
        camera_height=1.0, safety_radius=0.4, lambda_safety=5.0, lambda_img=2.0
    ) -> Dict:
        dist_pred, action_pred, action_scores, z_obs, z_goal = self.get_feature(obs_img, goal_img)
        
        # 1. 物理避障惩罚
        collision_costs = self.safety_loss_fn(
            action_pred[..., :2], depth_img, intrinsics, 
            camera_height=camera_height, safety_radius=safety_radius
        )

        # 2. 脑补潜力得分
        img_scores = self.evaluate_imagination(z_obs, z_goal, action_pred)

        # 3. 最终加权：专家偏好 - 安全惩罚 + 脑补加成
        final_scores = action_scores - lambda_safety * collision_costs + lambda_img * img_scores
        
        best_idx = torch.argmax(final_scores, dim=1)
        gather_idx = best_idx.view(-1, 1, 1, 1).expand(-1, -1, action_pred.shape[2], action_pred.shape[3])
        
        return {
            'best_action': torch.gather(action_pred, 1, gather_idx).squeeze(1),
            'dist_pred': dist_pred,
            'final_scores': final_scores
        }

    def forward(self, obs_img, goal_img, action_label=None, z_seq=None) -> Dict:
        """
        训练时: z_seq 为通过 obs_encoder 预先提取的完整序列特征
        """
        dist_pred, action_pred, action_scores, z_obs, z_goal = self.get_feature(obs_img, goal_img)
        
        loss_dyn = None
        if action_label is not None and z_seq is not None:
            loss_dyn = self.train_dynamics_step(z_seq, action_label)

        return {
            'dist_pred': dist_pred, 'action_pred': action_pred, 
            'action_scores': action_scores, 'loss_dyn': loss_dyn
        }

# --- TTA 处理器 ---
class TTAHandler:
    def __init__(self, dynamics_model, action_embed, lr=1e-5):
        self.model = dynamics_model
        self.action_embed = action_embed
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.buffer = []

    def update(self, z_t, a_real, z_next):
        self.buffer.append((z_t.detach(), a_real.detach(), z_next.detach()))
        if len(self.buffer) > 64: self.buffer.pop(0)
        if len(self.buffer) >= 16:
            batch = random.sample(self.buffer, 16)
            b_zt, b_at, b_zgt = [torch.stack([x[i] for x in batch]) for i in range(3)]
            self.model.train()
            z_pred = self.model(b_zt, self.action_embed(b_at))
            loss = (1 - F.cosine_similarity(z_pred, b_zgt, dim=-1)).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.eval()
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict, Optional, Tuple
import torch.utils
import torch.utils.checkpoint
from vint_train.models.lsnet.lsnet import lsnet_t, lsnet_b, lsnet_s
from vint_train.models.base_model import BaseModel

MODEL_DICT = {
    "t": lsnet_t,
    "s": lsnet_s,
    "b": lsnet_b,
}

class TemporalMemoryModule(nn.Module):
    """
    带时间位置编码 + 显式时间衰减bias + 多slot记忆的时序模块：
    - hist 加时间位置编码
    - 最近帧通过显式权重更重要
    - 多个 memory slots 跨时间累计信息（slot attention 风格）
    - 当前帧 feat_curr 对 [memory slots + 当前 hist] 做 attention，输出 feat_state
    """
    def __init__(
            self,
            d_model: int,
            num_heads: int = 8,
            max_hist_len: int = 5,
            num_slots: int = 4
        ):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots

        self.time_embed = nn.Embedding(max_hist_len, d_model)
        self.hist_decay_logit = nn.Parameter(torch.tensor(0.0))
        self.slot_attn = nn.MultiheadAttention(
            d_model, num_heads=num_heads, batch_first=True
        )
        self.slot_norm = nn.LayerNorm(d_model)

        self.W_z = nn.Linear(2 * d_model, d_model)
        self.W_r = nn.Linear(2 * d_model, d_model)
        self.W_h = nn.Linear(2 * d_model, d_model)
        self.decay_logit = nn.Parameter(torch.tensor(0.0))

        self.mem_k_proj = nn.Linear(d_model, d_model)
        self.mem_v_proj = nn.Linear(d_model, d_model)

        self.temporal_attn = nn.MultiheadAttention(
            d_model, num_heads=num_heads, batch_first=True
        )
        self.temporal_norm = nn.LayerNorm(d_model)
        self.slot_init = nn.Parameter(torch.randn(1, num_slots, d_model))

    def init_memory(self, batch_size: int, device: torch.device):
        return self.slot_init.expand(batch_size, -1, -1).to(device)
    
    def forward(
        self,
        feat_curr: torch.Tensor,
        feat_hist: torch.Tensor,
        mem_prev: Optional[torch.Tensor] = None,
    ):
        B, T, _ = feat_hist.shape
        device = feat_hist.device

        time_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        time_emb = self.time_embed(time_ids)
        feat_hist_pos = feat_hist + time_emb

        dist_rev = (T - 1) - torch.arange(T, device=device)
        lam = F.softplus(self.hist_decay_logit)
        time_weights = torch.exp(-lam * dist_rev.float())
        time_weights = time_weights / time_weights.max()
        feat_hist_pos = feat_hist_pos * time_weights.view(1, T, 1)
        if mem_prev is None:
            slots = self.init_memory(B, device)
        else:
            slots = mem_prev

        attn_out, _ = self.slot_attn(
            query=slots, key=feat_hist_pos, value=feat_hist_pos
        )
        slot_candidate = self.slot_norm(slots + attn_out)

        mem_decay_factor = torch.sigmoid(self.decay_logit)
        mem_decay = mem_decay_factor * slots

        x = torch.cat([mem_decay, slot_candidate], dim=-1)
        z = torch.sigmoid(self.W_z(x))
        r = torch.sigmoid(self.W_r(x))
        h_tilde = torch.tanh(
            self.W_h(torch.cat([r * mem_decay, slot_candidate], dim=-1))
        )
        slots = (1.0 - z) * mem_decay + z * h_tilde

        mem_next = slots

        K_mem = self.mem_k_proj(mem_next)
        V_mem = self.mem_v_proj(mem_next)

        K_all = torch.cat([K_mem, feat_hist_pos], dim=1)
        V_all = torch.cat([V_mem, feat_hist_pos], dim=1)

        q = feat_curr.unsqueeze(1)
        attn_out, _ = self.temporal_attn(q, K_all, V_all)
        feat_state = self.temporal_norm(q + attn_out).squeeze(1)

        return feat_state, mem_next

    
class LSNet(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoding_size: Optional[int] = 1024,
        goal_encoding_size: Optional[int] = 1024,
        pretrained: Optional[bool] = False,
        model_size: Optional[str] = 't',
        img_size: Optional[Tuple[int, int]] = [224, 224],
        depth_img: Optional[bool] = False
    ) -> None:
        """
        GNM main class
        Args:
            context_size (int): how many previous observations to used for context
            len_traj_pred (int): how many waypoints to predict in the future
            learn_angle (bool): whether to predict the yaw of the robot
            obs_encoding_size (int): size of the encoding of the observation images
            goal_encoding_size (int): size of the encoding of the goal images
        """
        super(LSNet, self).__init__(context_size, len_traj_pred, learn_angle)
        self.context_size = context_size
        self.depth_img = depth_img
        embed_dim = 384 if model_size == 't' else 448 if model_size == 's' else 512
        in_chans_obs = 4 if depth_img else 3
        # in_chans_goal = 8 if depth_img else 6
        in_chans_goal = 6
        if depth_img:
            self.obs_depthBlock = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=3, padding=1),
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True)
            )
            self.goal_depthBlock = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=3, padding=1),
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True)
            )
        self.obs_encoder = MODEL_DICT[model_size](pretrained=pretrained, img_size=img_size[0], in_chans=in_chans_obs)
        self.goal_encoder = MODEL_DICT[model_size](pretrained=pretrained, img_size=img_size[0], in_chans=in_chans_goal)
        
        # self.temporal_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        # self.temporal_norm = nn.LayerNorm(embed_dim)
        self.type_embed = nn.Embedding(3, embed_dim)
        self.temporal_memory = TemporalMemoryModule(
            d_model=embed_dim,
            num_heads=8,
            max_hist_len=context_size,
            num_slots=4
        )
        self.memory: Optional[torch.Tensor] = None


        self.goal_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.goal_norm = nn.LayerNorm(embed_dim)

        self.linear_layers = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
        )
        self.dist_predictor = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        self.action_predictor = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, self.len_trajectory_pred * self.num_action_params),
        )
        self.lfp_predictor = nn.Sequential(
            nn.Linear(embed_dim, 512), 
            nn.GELU(),
            nn.Linear(512, embed_dim)
        )
        self.state_proj = nn.Linear(embed_dim * 2, embed_dim)

    def reset_memory(self):
        self.memory = None

    def forward(
        self, obs_img: torch.tensor, goal_img: torch.tensor, 
        obs_img_depth: torch.tensor, goal_img_depth: torch.tensor, 
        next_img: torch.tensor, next_img_depth: torch.tensor,
        memory: Optional[torch.Tensor] = None
        # obs_img: torch.Size([400, 18, 64, 85])    goal_img: torch.Size([400, 3, 64, 85])
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 预处理三个特征
        B, _, H, W = obs_img.shape
        hist_img = obs_img[:, :-3, :, :]
        hist_flat = hist_img.reshape(-1, 3, H, W)
        curr_img = obs_img[:, -3:, :, :]
        curr_img_ = curr_img.clone()
        
        if self.depth_img:
            hist_img_depth = obs_img_depth[:, :-1, :, :]
            hist_flat_depth = hist_img_depth.reshape(-1, 1, H, W)
            hist_flat_depth_in = self.obs_depthBlock(hist_flat_depth)
            curr_img_depth = obs_img_depth[:, -1:, :, :]
            curr_img_depth_in = self.obs_depthBlock(curr_img_depth)
            # goal_img_depth_in = self.goal_depthBlock(goal_img_depth)
            curr_img = torch.cat([curr_img, curr_img_depth_in], dim=1)
            hist_flat = torch.cat([hist_flat, hist_flat_depth_in], dim=1)
            # goal_img = torch.cat([goal_img, goal_img_depth_in], dim=1)
            with torch.no_grad():
                next_img_depth_in = self.obs_depthBlock(next_img_depth)
                next_img = torch.cat([next_img, next_img_depth_in], dim=1)
        # 三个特征同时进行视觉编码
        feat_curr = self.obs_encoder(curr_img)
        feat_hist = self.obs_encoder(hist_flat)
        feat_hist = feat_hist.reshape(B, -1, feat_curr.shape[-1])
        feat_goal = self.goal_encoder(torch.cat([curr_img_, goal_img], dim=1))
        # curr来query hist，回顾历史
        # q_temp = feat_curr.unsqueeze(1)
        # attn_out, _ = self.temporal_attn(query=q_temp, key=feat_hist, value=feat_hist)
        # feat_state = self.temporal_norm(q_temp + attn_out).squeeze(1)
        device = obs_img.device
        type_curr = self.type_embed(torch.tensor(0, device=device))
        type_hist = self.type_embed(torch.tensor(1, device=device))
        type_goal = self.type_embed(torch.tensor(2, device=device))
        feat_curr = feat_curr + type_curr.unsqueeze(0)
        feat_hist = feat_hist + type_hist.view(1, 1, -1)
        feat_goal = feat_goal + type_goal.unsqueeze(0)

        mem_prev = self.memory
        if mem_prev is not None and mem_prev.size(0) != B:
            mem_prev = None
        feat_state, mem_next = self.temporal_memory(
            feat_curr=feat_curr,
            feat_hist=feat_hist,
            mem_prev=mem_prev
        )
        self.memory = mem_next

        with torch.no_grad():
            feat_next = self.obs_encoder(next_img)
        pred_next_latent = self.lfp_predictor(feat_state)
        lfp_loss = F.mse_loss(pred_next_latent, feat_next)
        
        # 融合预测帧
        feat_state = self.state_proj(torch.cat([feat_state, pred_next_latent], dim=-1))
        # goal进行fusion
        q_goal = feat_goal.unsqueeze(1)
        k_goal = feat_state.unsqueeze(1)
        attn_goal, _ = self.goal_attn(query=q_goal, key=k_goal, value=k_goal)
        feat_goal_attended = self.goal_norm(q_goal + attn_goal).squeeze(1)

        combined = torch.cat([feat_goal_attended, feat_goal], dim=-1)

        z = self.linear_layers(combined)
        dist_pred = self.dist_predictor(z)
        action_pred = self.action_predictor(z)

        # augment outputs to match labels size-wise
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )
        action_pred[:, :, :2] = torch.cumsum(
            action_pred[:, :, :2], dim=1
        )  # convert position deltas into waypoints
        if self.learn_angle:
            action_pred[:, :, 2:] = F.normalize(
                action_pred[:, :, 2:].clone(), dim=-1
            )  # normalize the angle prediction
        return dist_pred, action_pred, lfp_loss, mem_next
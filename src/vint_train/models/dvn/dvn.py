import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from efficientnet_pytorch import EfficientNet
from vint_train.models.base_model import BaseModel
from vint_train.models.dvn.transformer_module import TransformerModule
from vint_train.models.dvn.heads import MultiHypothesisHead, CandidateConditionedScorer
from vint_train.models.dvn.dynamics import RobustDynamics
from vint_train.models.dvn.nav_module import TokenPooler
from vint_train.models.regularizers import SIGReg


class DVN(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        vision_encoder: Optional[str] = "efficientnet-b0",
        encoding_size: Optional[int] = 512,
        late_fusion: Optional[bool] = False,
        mha_num_attention_heads: Optional[int] = 4,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
        num_hypotheses: int = 5,
        sigreg_knots: int = 17,
        sigreg_num_proj: int = 1024,
        use_world_model: bool = True,
    ) -> None:

        super(DVN, self).__init__(context_size, len_traj_pred, learn_angle)
        self.num_action_params = 4 if learn_angle else 2
        self.len_traj_pred = len_traj_pred
        self.encoding_size = encoding_size
        self.action_embed_dim = 128
        self.policy_latent_dim = 128
        self.late_fusion = late_fusion
        # Keep the dynamics parameters in the module for parameter-matched
        # ablations, while allowing a clean no-imagined-endpoint baseline.
        self.use_world_model = use_world_model

        if not vision_encoder.startswith("efficientnet"):
            raise ValueError(
                f"DVN currently supports EfficientNet encoders only, got {vision_encoder!r}. "
                "Use obs_encoder: efficientnet-b0 for the stable DVN baseline."
            )

        self.obs_encoder = EfficientNet.from_name(vision_encoder, in_channels=3)
        self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=3 if late_fusion else 6)
        vision_emb = self.obs_encoder._fc.in_features

        self.compress_obs_enc = nn.Linear(vision_emb, self.encoding_size) if vision_emb != self.encoding_size else nn.Identity()
        self.compress_goal_enc = nn.Linear(vision_emb, self.encoding_size) if vision_emb != self.encoding_size else nn.Identity()

        # 0: History, 1: Observation, 2: Goal
        self.type_embed = nn.Embedding(3, self.encoding_size)
        self.seq_len = (context_size + 1) + 1

        self.transformer = TransformerModule(
            embed_dim=self.encoding_size,
            seq_len=self.seq_len,
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor
        )

        self.action_predictor = MultiHypothesisHead(
            input_dim=self.encoding_size,
            num_hypotheses=num_hypotheses,
            len_traj_pred=len_traj_pred,
            action_dim=self.num_action_params,
        )

        self.candidate_scorer = CandidateConditionedScorer(
            state_dim=self.encoding_size,
            traj_len=len_traj_pred,
            action_dim=self.num_action_params,
            traj_embed_dim=128,
            hidden_dim=256,
        )

        self.dist_predictor = nn.Sequential(
            nn.Linear(self.encoding_size, self.encoding_size // 4),
            nn.LayerNorm(self.encoding_size // 4),
            nn.GELU(),
            nn.Linear(self.encoding_size // 4, self.encoding_size // 16),
            nn.GELU(),
            nn.Linear(self.encoding_size // 16, 1)
        )

        self.pool = TokenPooler()
                
        self.action_embed = nn.Sequential(
            nn.Linear(self.num_action_params, self.action_embed_dim),
            nn.GELU(),
            nn.Linear(self.action_embed_dim, self.action_embed_dim),
        )

        self.dynamics = RobustDynamics(self.encoding_size, self.action_embed_dim, hidden_dim=512)
        self.dyn_norm = nn.LayerNorm(self.encoding_size)
        self.sigreg = SIGReg(knots=sigreg_knots, num_proj=sigreg_num_proj)

    def get_feature(
        self, 
        obs_img: torch.Tensor,
        goal_img: torch.Tensor,
        next_img: Optional[torch.Tensor] = None,
    ):
        device = obs_img.device
        B, S, C, H, W = obs_img.shape
        expected_obs_len = self.context_size + 1
        if S != expected_obs_len:
            raise ValueError(f"DVN expects {expected_obs_len} observation frames, got {S}")

        curr_img = obs_img[:, -1]
        obs_flat = obs_img.reshape(B * S, C, H, W)
        obs_tokens = self.obs_encoder.extract_features(obs_flat)
        obs_feat_seq = self.compress_obs_enc(self.obs_encoder._avg_pooling(obs_tokens).flatten(1))
        obs_feat_seq = obs_feat_seq.reshape(B, S, self.encoding_size)
        hist_feat = obs_feat_seq[:, :self.context_size]
        obs_feat = obs_feat_seq[:, self.context_size:self.context_size + 1]

        next_feat = None
        sigreg_inputs = obs_feat_seq
        if next_img is not None:
            Bn, Sn, Cn, Hn, Wn = next_img.shape
            if Bn != B or Cn != C:
                raise ValueError(
                    f"next_img shape {tuple(next_img.shape)} is incompatible with obs_img {tuple(obs_img.shape)}"
                )
            next_flat = next_img.reshape(Bn * Sn, Cn, Hn, Wn)
            next_tokens = self.obs_encoder.extract_features(next_flat)
            next_feat = self.compress_obs_enc(self.obs_encoder._avg_pooling(next_tokens).flatten(1))
            next_feat = next_feat.reshape(B, Sn, self.encoding_size)
            sigreg_inputs = torch.cat([obs_feat_seq, next_feat], dim=1)

        goal_feat = self.goal_encoder.extract_features(torch.cat([curr_img, goal_img], dim=1))
        goal_feat = self.compress_goal_enc(self.goal_encoder._avg_pooling(goal_feat).flatten(1)).unsqueeze(1)       # torch.Size([B, 1, 512])
        
        sigreg_loss = self.sigreg(sigreg_inputs)

        transformer_tokens = torch.cat([hist_feat, obs_feat, goal_feat], dim=1)
        t_hist = torch.full((B, self.context_size), 0, device=device)
        t_obs  = torch.full((B, 1), 1, device=device)
        t_goal = torch.full((B, 1), 2, device=device)
        type_ids = torch.cat([t_hist, t_obs, t_goal], dim=1)
        transformer_tokens = transformer_tokens + self.type_embed(type_ids)

        transformer_out = self.transformer(transformer_tokens)
        z_obs = transformer_out[:, -2, :]
        z_goal = transformer_out[:, -1, :]
        pooled_out = transformer_out.mean(dim=1)

        dist_pred = self.dist_predictor(pooled_out)
        action_pred = self.action_predictor(pooled_out)

        action_pred[..., :2] = torch.cumsum(action_pred[..., :2], dim=2)
        if self.learn_angle:
            action_pred[..., 2:] = F.normalize(action_pred[..., 2:].clone(), dim=-1)

        return dist_pred, action_pred, next_feat, obs_feat, z_obs, z_goal, sigreg_loss

    def evaluate_imagination(
        self,
        z_t: torch.Tensor,
        z_goal: torch.Tensor,
        action_pred: torch.Tensor,
    ) -> torch.Tensor:
        B, K, T, _ = action_pred.shape
        curr_z = z_t.unsqueeze(1).repeat(1, K, 1).reshape(B * K, -1)
        z_goal_flat = z_goal.unsqueeze(1).repeat(1, K, 1).reshape(B * K, -1)

        for t in range(T):
            a_t = action_pred[:, :, t, :].reshape(B * K, -1)
            a_emb = self.action_embed(a_t)
            curr_z = self.dynamics(curr_z, a_emb)
            curr_z = self.dyn_norm(curr_z)

        z_end = curr_z.view(B, K, -1)
        img_scores = F.cosine_similarity(
            F.normalize(curr_z, dim=-1),
            F.normalize(z_goal_flat, dim=-1),
            dim=-1,
        ).view(B, K)
        return img_scores, z_end
    
    def train_dynamics_step(self, action_pred, action_label, next_feat, next_mask, obs_feat, epoch):
        B, L, D = next_feat.shape
        if L <= 0:
            return torch.tensor(0.0).to(next_feat.device)
        if next_mask is None:
            next_mask = torch.ones((B, L), device=next_feat.device)
        else:
            next_mask = next_mask.to(next_feat.device)
        
        curr_z = obs_feat.squeeze(1)
        total_loss = torch.tensor(0.0, device=next_feat.device)
        valid_steps_count = 0

        for h in range(L):
            m_t = next_mask[:, h]
            if m_t.sum() == 0:
                continue
            a_t = action_label[:, h, :]
            a_emb = self.action_embed(a_t)
            curr_z = self.dynamics(curr_z, a_emb)
            curr_z = self.dyn_norm(curr_z)
            z_gt = next_feat[:, h, :].detach()
            cos_sim = F.cosine_similarity(curr_z, z_gt, dim=-1)
            step_loss_per_sample = 1 - cos_sim
            masked_step_loss = (step_loss_per_sample * m_t).sum()
            total_loss += masked_step_loss
            valid_steps_count += m_t.sum()

        return total_loss / (valid_steps_count + 1e-6)

    def forward(
        self, 
        obs_img: torch.Tensor,
        goal_img: torch.Tensor,
        action_label: Optional[torch.Tensor] = None,
        next_img: Optional[torch.Tensor] = None,
        next_mask: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
    ) -> dict:
        dist_pred, action_pred, next_feat, obs_feat, z_obs, z_goal, sigreg_loss = self.get_feature(obs_img, goal_img, next_img)

        if self.use_world_model:
            _, z_end = self.evaluate_imagination(z_obs, z_goal, action_pred)
        else:
            # The ablation keeps the scorer input shape and parameter count
            # unchanged, but removes the imagined future state entirely.
            z_end = z_obs.unsqueeze(1).expand(-1, action_pred.shape[1], -1)
        action_scores = self.candidate_scorer(
            z_obs=z_obs,
            z_goal=z_goal,
            z_end=z_end,
            trajs=action_pred,
        )

        loss_dyn = torch.zeros((), device=z_obs.device)
        if self.use_world_model and action_label is not None and next_feat is not None:
            loss_dyn = self.train_dynamics_step(action_pred, action_label, next_feat, next_mask, obs_feat, epoch)

        outputs = {
            'dist_pred': dist_pred, 
            'action_pred': action_pred, 
            'action_scores': action_scores,
            'loss_dyn': loss_dyn,
            'sigreg_loss': sigreg_loss,
        }

        return outputs

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from vint_train.models.base_model import BaseModel
from vint_train.models.dinov2.adapter import BottleneckAdapter
from vint_train.models.dinov2.depth_encoder import DepthEncoder
from vint_train.models.dinov2.heads import MultiHypothesisHead
from vint_train.models.dinov2.attention_pooling import DepthMultiQueryPooling
from vint_train.models.dinov2.transformer_module import TransformerModule

from vint_train.models.dinov2.safety_loss import DifferentiableCollisionLoss

class DINOv2RGBDPolicy(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoding_size: Optional[int] = 768, # Default for ViT-B/14
        mha_num_attention_heads: Optional[int] = 4,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
        num_hypotheses: int = 5,
    ) -> None:
        """
        Args:
            context_size: Number of past frames
            obs_encoding_size: DINOv2 embedding size (768 for ViT-B)
            num_hypotheses: Number of trajectory hypotheses for MHP
        """
        super().__init__(context_size, len_traj_pred, learn_angle)
        # Explicitly define num_action_params to ensure it's correct regardless of BaseModel
        self.num_action_params = 4 if learn_angle else 2
        
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        policy_latent_dim = 128

        # 1. RGB Backbone: DINOv2 (Frozen)
        print("Loading DINOv2 Backbone...")
        hub_dir = os.path.expanduser('~/.cache/torch/hub/facebookresearch_dinov2_main')
        self.rgb_backbone = torch.hub.load(hub_dir, 'dinov2_vitb14', source='local')
        # self.rgb_backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            
        # 2. Adapters for RGB (Trainable)
        # Split Obs/Goal to handle different semantic roles
        # self.obs_adapter = BottleneckAdapter(in_features=obs_encoding_size * 2, bottleneck_dim=256)
        # self.goal_adapter = BottleneckAdapter(in_features=obs_encoding_size, bottleneck_dim=256)
        
        # 3. Depth Backbone (Trainable)
        self.depth_encoder = DepthEncoder(output_dim=obs_encoding_size, backbone_name='resnet18')
        
        # 3.1 Depth Aggregator (DepthMultiQueryPooling)
        # Using 4 queries (Front, Left, Right, Center) with spatial priors
        self.num_depth_queries = 4
        self.depth_aggregator = DepthMultiQueryPooling(
            embed_dim=obs_encoding_size, 
            num_queries=self.num_depth_queries, 
            num_heads=4
        )
        
        # 4. Modality Fusion Layer
        # Fuses RGB (768) + Depth (4 * 768) -> Fused (768)
        # We flatten the Q depth tokens: 4 * 768 = 3072 input features
        fusion_input_dim = obs_encoding_size + (self.num_depth_queries * obs_encoding_size)
        
        # 2-Layer MLP Fusion (Enhanced)
        self.modality_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, obs_encoding_size),
            nn.LayerNorm(obs_encoding_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(obs_encoding_size, obs_encoding_size),
            nn.Dropout(0.1) # No residual, let network learn fusion
        )
        
        # 5. Type Embeddings
        # 0: Observation, 1: Goal
        self.type_embed = nn.Embedding(2, obs_encoding_size)
        
        # Sequence: [Fused_0, Fused_1, ..., Fused_Context, Goal_RGB]
        # Length = (Context + 1) + 1
        self.seq_len = (context_size + 1) + 1
        
        # 6. Transform & Encoding
        # Using extracted TransformerModule with Positional Encoding
        self.transformer = TransformerModule(
            embed_dim=self.obs_encoding_size,
            seq_len=self.seq_len,
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor
        )
        
        # Policy Head Compressor (Replaces MultiLayerDecoder's MLP part)
        # Takes Goal Token (Pooled Repr) -> 32 dim latent
        # Input dim is just obs_encoding_size (768), not SeqLen*768
        self.policy_compressor = nn.Sequential(
            nn.LayerNorm(obs_encoding_size),
            nn.Linear(obs_encoding_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, policy_latent_dim),
        )

        self.dist_predictor = nn.Sequential(
            nn.Linear(obs_encoding_size * 4, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            # nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Action Predictor (MHP)
        # Removed duplicate initialization. Using num_action_params logic defined in __init__
        self.action_predictor = MultiHypothesisHead(
            input_dim=policy_latent_dim,
            num_hypotheses=num_hypotheses,
            len_traj_pred=len_traj_pred,
            action_dim=self.num_action_params 
        )
        
        # 8. Action-Conditioned Dynamics Model (LFP Upgrade)
        # Replacing simple LFP head with Dynamics: z_next = f(z_curr, a)
        # Note: We use last_obs_token as z_curr (768 dim).
        self.action_embed_dim = 128
        self.action_embed = nn.Sequential(
            nn.Linear(self.num_action_params, self.action_embed_dim),
            nn.GELU(),
            nn.Linear(self.action_embed_dim, self.action_embed_dim),
        )

        self.dynamics = nn.Sequential(
            nn.Linear(obs_encoding_size + self.action_embed_dim, 512),
            nn.GELU(),
            nn.Linear(512, obs_encoding_size),
        )
        self.dyn_norm = nn.LayerNorm(obs_encoding_size)
        
        # 9. Safety Loss
        self.safety_loss_fn = DifferentiableCollisionLoss(safety_radius=0.5)

    def train(self, mode: bool = True):
        super().train(mode)
        # Force backbone to eval mode even when model is in train mode
        # self.rgb_backbone.eval()

    def unfreeze_backbone(self, num_blocks: int = 4):
        """
        Unfreeze the last num_blocks of the backbone and the final norm layer.
        """
        for p in self.rgb_backbone.parameters():
            p.requires_grad = False

        if hasattr(self.rgb_backbone, 'head'):
             for p in self.rgb_backbone.head.parameters():
                 p.requires_grad = True
        
        if hasattr(self.rgb_backbone, 'blocks'):
            total_blocks = len(self.rgb_backbone.blocks)
            to_unfreeze = self.rgb_backbone.blocks[total_blocks - num_blocks :]
            
            for block in to_unfreeze:
                for p in block.parameters():
                    p.requires_grad = True
        # for norm_name in ["norm", "fc_norm"]:
        #     if hasattr(self.rgb_backbone, norm_name):
        #         for p in getattr(self.rgb_backbone, norm_name).parameters():
        #             p.requires_grad = True
        print(f"DINOv2: Unfroze last {num_blocks} blocks and final norm.")

    def forward(
        self, 
        obs_img: torch.Tensor, 
        goal_img: torch.Tensor, 
        depth_img: torch.Tensor,
        action: Optional[torch.Tensor] = None, # [B, T, A] or [B, A] depending on horizon. Usually just next action [B, A]
        next_obs_img: Optional[torch.Tensor] = None # For training dynamics
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        device = obs_img.device
        B = obs_img.shape[0]

        # --- 1. Process RGB Observations ---
        # obs_img: [B, 3*(Context+1), H, W]
        # Check standard input format. usually 3*Context. 
        # But wait, ViNT code splits by 3.
        # Let's assume input is concatenated channel-wise.
        
        # Reshape to [B*(Context+1), 3, H, W] for batch processing
        obs_img_list = torch.split(obs_img, 3, dim=1)
        obs_img_batch = torch.cat(obs_img_list, dim=0) 
        
        # with torch.no_grad():
            # DINOv2 Forward
            # output depends on model. Using forward_features for control or just forward for CLS
            # hub model forward returns features dict or cls? 
            # dinov2_vitb14 forward returns the CLS token + Patch tokens? 
            # Actually standard hub model forward() returns the output of the last block.
            # let's use forward_features() to be safe if available, or just forward
        rgb_feats_full = self.rgb_backbone.forward_features(obs_img_batch)
        obs_patch = rgb_feats_full['x_norm_patchtokens'] # [B_total, N, 768] x_norm_clstoken
        # obs_cls = rgb_feats_full["x_norm_clstoken"]
        obs_patch_mean = obs_patch.mean(dim=1)
        # obs_combined = torch.cat([obs_cls, obs_patch_mean], dim=-1)
            
        # Adapt Obs
        # rgb_tokens = self.obs_adapter(obs_combined) # [B_total, 768]
        rgb_tokens = obs_patch_mean.clone()
        rgb_tokens = rgb_tokens.view(B, -1, self.obs_encoding_size) # [B, Context+1, 768]
        
        # --- 2. Process RGB Goal ---
        # with torch.no_grad():
        goal_feats_full = self.rgb_backbone.forward_features(goal_img)
        # Use goal patches, NOT obs patches
        goal_patch = goal_feats_full['x_norm_patchtokens'] # [B_total, N, 768]
        # goal_cls = goal_feats_full["x_norm_clstoken"]
        goal_patch_mean = goal_patch.mean(dim=1)
        # goal_combined = torch.cat([goal_cls, goal_patch_mean], dim=-1)
            
        # Adapt Goal
        # goal_tokens = self.obs_adapter(goal_combined) # [B, 768]
        goal_tokens = goal_patch_mean.clone()
        goal_tokens = goal_tokens.unsqueeze(1)   # [B, 1, 768]

        # --- 3. Process Depth ---
        # depth_img: [B, 1*(Context+1), H, W]
        depth_img_list = torch.split(depth_img, 1, dim=1)
        depth_img_batch = torch.cat(depth_img_list, dim=0) # [B_total, 1, H, W]
        
        depth_spatial = self.depth_encoder(depth_img_batch) # [B_total, N, 768]
        
        # Pool depth tokens using Multi-Query Pooling
        # depth_spatial: [B_total, N, D] (N=49)
        # -> depth_tokens: [B_total, Q, D] (Q=4)
        # We need to pass grid_hw for spatial priors. ResNet18 strice 32 -> 224/32 = 7.
        depth_tokens = self.depth_aggregator(depth_spatial, grid_hw=(7, 7)) 
        
        # Flatten Q tokens -> [B_total, Q*D]
        depth_tokens = depth_tokens.flatten(1) # [B_total, 3072]
        
        depth_tokens = depth_tokens.view(B, -1, self.num_depth_queries * self.obs_encoding_size) # [B, Context+1, 3072]
        
        # --- 4. Modality Fusion (Per Frame) ---
        # rgb_tokens: [B, T, D], depth_tokens: [B, T, D]
        # Concat along dim=-1 (Feature dim) -> [B, T, 2D]
        fused_obs_tokens = torch.cat([rgb_tokens, depth_tokens], dim=-1)
        fused_obs_tokens = self.modality_fusion(fused_obs_tokens) # [B, T, D]
        
        # Append Goal
        final_seq = torch.cat([fused_obs_tokens, goal_tokens], dim=1) # [B, T+1, 768]
        
        # Add Type Embeddings
        # 0 for Obs (first T tokens), 1 for Goal (last 1 token)
        T_obs = fused_obs_tokens.shape[1]
        type_ids = torch.zeros((B, final_seq.shape[1]), dtype=torch.long, device=device)
        type_ids[:, T_obs:] = 1 # Set last token to Goal Type
        
        final_seq = final_seq + self.type_embed(type_ids)
        
        # --- 5. Transformer Encoding ---
        # Output: [B, SeqLen, D]
        transformer_out = self.transformer(final_seq)
        
        # --- 6. Dual Heads Branching ---
        
        # Batch A: Policy Branch
        # Use Goal Token (Last Token) as the global representation for policy
        # transformer_out: [B, SeqLen, D] -> [B, D]
        goal_token_repr = transformer_out[:, -1, :] 
        policy_latent = self.policy_compressor(goal_token_repr) # [B, 128]
        
        # --- 7. Policy Predictions ---
        z_obs  = transformer_out[:, -2, :]
        # z_obs_norm = F.normalize(z_obs, p=2, dim=-1)
        # z_goal_norm = F.normalize(goal_token_repr, p=2, dim=-1)
        # sim = F.cosine_similarity(z_obs_norm, z_goal_norm)
        # print('sim', sim)
        # dot_product = torch.dot(z_obs_norm.view(-1), z_goal_norm.view(-1))
        # print('dot_product', dot_product)
        # dist = torch.dist(z_obs_norm, z_goal_norm, p=2)
        # print('dist', dist)
        # dist_in = torch.cat([z_obs_norm, z_goal_norm, z_goal_norm - z_obs_norm, z_goal_norm * z_obs_norm], dim=-1)
        dist_in = torch.cat([z_obs, goal_token_repr, goal_token_repr - z_obs, goal_token_repr * z_obs], dim=-1)
        dist_pred = self.dist_predictor(dist_in)
        
        # MHP Forward
        action_pred, action_scores = self.action_predictor(policy_latent) 
        # action_pred shape: [B, K, T, A]
        
        # Waypoint cumsum (Apply to all K hypotheses)
        # action_pred[:, :, :, :2] is [B, K, T, 2]
        xy = torch.cumsum(action_pred[..., :2], dim=2)
        action_pred = torch.cat([xy, action_pred[..., 2:]], dim=-1)

        
        if self.learn_angle:
            action_pred[:, :, :, 2:] = F.normalize(action_pred[:, :, :, 2:].clone(), dim=-1)

        # --- 8. Dynamics / LFP (Conditional) ---
        loss_dyn = None
        
        # Branch B: Dynamics Branch input
        # User Request: LFP loss should optimize Transformer + Dynamics, but NOT Encoders.
        # Solution: Run Transformer again with detached input for LFP branch.
        # This allows gradients to flow into Transformer weights, but stops at final_seq.
        
        if action is not None and next_obs_img is not None:
            # 1. Detach input to stop gradient flow to encoders
            final_seq_detached = final_seq.detach()
            
            # 2. Re-run Transformer (Cheap compared to Backbone)
            transformer_out_dyn = self.transformer(final_seq_detached)
            z_t = transformer_out_dyn[:, -2, :] # [B, 768]
            
            # Training Mode: Calculate Dynamics Loss
            # action: [B, A] (The action taken at time t)
            a_t = action 
            a_emb = self.action_embed(a_t) # [B, 128]
            
            # Dynamics Forward
            z_pred_next = self.dynamics(torch.cat([z_t, a_emb], dim=-1)) # [B, 768]
            z_pred_next = self.dyn_norm(z_pred_next)
            
            # Get GT Target
            #  with torch.no_grad():
            # Process next_obs_img similarly to obs_img
            # Assuming next_obs_img is [B, 3, H, W] (Single frame)
            feat_next_full = self.rgb_backbone.forward_features(next_obs_img)
            # feat_next_cls = feat_next_full['x_norm_clstoken'] # [B, 768]
            feat_next_patch = feat_next_full['x_norm_patchtokens']
            feat_next_patch_mean = feat_next_patch.mean(dim=1)
            # feat_next_cls_combined = torch.cat([feat_next_cls, feat_next_patch_mean], dim=-1)

            
            # Adapt Target using the SAME adapter as inputs (shared embedding space)
            # NOTE: We detached input to Transformer, but z_next_tgt comes from Adapter.
            # If we want to freeze Adapter for LFP task, we should detach here too?
            # But 'obs_adapter' *is* the encoder.
            # Z_tgt is the ground truth feature. We probably DO want to adapt it?
            # User said: "not optimize depth_encoder ...". 
            # Standard LFP: usually assumes fixed target or target from Momentum Encoder.
            # Here we use the online encoder.
            # Let's detach the target completely to avoid "Collapsing" solutions?
            # Or just allow Adapter to learn "predictability"?
            # Given "not optimize ... other modules", detach target is saftest.
            # z_next_tgt = self.obs_adapter(feat_next_cls_combined).detach() # [B, 768]
            z_next_tgt = feat_next_patch_mean.clone().detach()


        # Cosine Similarity Loss (Maximize similarity)
            z_pred = F.normalize(z_pred_next, dim=-1)
            z_tgt = F.normalize(z_next_tgt, dim=-1)
            loss_dyn = (1 - (z_pred * z_tgt).sum(dim=-1)).mean()
             
        # 9. Safety Loss (Collision)
        # Calculate for all hypotheses.
        # depth_img: Current depth frame (usually index -1 if stacked).
        # We need single frame depth: [B, 1, H, W]
        # Our depth_img input is [B, Context+1, H, W] (from train_utils).
        # We take the *last* frame which corresponds to the current observation.
        depth_curr = torch.split(depth_img, 1, dim=1)[-1]
        
        intrinsics = self.safety_loss_fn.get_default_intrinsics(B, device)
        loss_collision = self.safety_loss_fn(action_pred[..., :2], depth_curr, intrinsics)
        
        return dist_pred, action_pred, action_scores, loss_dyn, loss_collision

    def predict_with_imagination(
        self,
        obs_img: torch.Tensor,
        goal_img: torch.Tensor,
        depth_img: torch.Tensor,
        num_samples: int = 5,
        w_consistency: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Advanced Inference: Uses Dynamics Model to re-rank action hypotheses.
        
        Args:
            obs_img, goal_img, depth_img: Standard inputs
            num_samples: Number of hypotheses to evaluate (should match num_hypotheses in train)
            w_consistency: Weight for dynamics consistency score (0.0 = only policy score, 1.0 = only dyn score)
            
        Returns:
            best_action: [B, T, A] The selected best trajectory
            scores: [B, K] The combined scores
        """
        device = obs_img.device
        B = obs_img.shape[0]
        
        # 1. Standard Forward Pass (No Action, No LFP Loss)
        # This gets us the policy hypotheses and the current embeddings
        # We need to access internal features, so we replicate some forward logic or modify forward to return them.
        # Ideally, we refactor forward to split feature extraction.
        # But for now, let's just copy the feature extraction part to ensure we get 'z_t' and 'goal_tokens'.
        
        # --- Feature Extraction (Copy of Forward) ---
        # (This is slightly inefficient code duplication but safer for not breaking training)
        obs_img_list = torch.split(obs_img, 3, dim=1)
        obs_img_batch = torch.cat(obs_img_list, dim=0) 
        # with torch.no_grad():
        rgb_feats_full = self.rgb_backbone.forward_features(obs_img_batch)
        rgb_cls = rgb_feats_full['x_norm_clstoken'] 
        rgb_tokens = self.obs_adapter(rgb_cls).view(B, -1, self.obs_encoding_size)

        # with torch.no_grad():
        goal_feats_full = self.rgb_backbone.forward_features(goal_img)
        goal_cls = goal_feats_full['x_norm_clstoken']
        goal_tokens = self.obs_adapter(goal_cls).unsqueeze(1)
        
        # We use a detached, normalized goal embedding as the target for "Goal Progress"
        z_goal = F.normalize(goal_tokens.squeeze(1), dim=-1) # [B, 768]

        depth_img_list = torch.split(depth_img, 1, dim=1)
        depth_img_batch = torch.cat(depth_img_list, dim=0)
        depth_spatial = self.depth_encoder(depth_img_batch)
        depth_tokens = self.depth_aggregator(depth_spatial, grid_hw=(7, 7)).flatten(1)
        depth_tokens = depth_tokens.view(B, -1, self.num_depth_queries * self.obs_encoding_size)
        
        fused_obs_tokens = torch.cat([rgb_tokens, depth_tokens], dim=-1)
        fused_obs_tokens = self.modality_fusion(fused_obs_tokens)
        
        final_seq = torch.cat([fused_obs_tokens, goal_tokens], dim=1)
        T_obs = fused_obs_tokens.shape[1]
        type_ids = torch.zeros((B, final_seq.shape[1]), dtype=torch.long, device=device)
        type_ids[:, T_obs:] = 1
        final_seq = final_seq + self.type_embed(type_ids)
        
        transformer_out = self.transformer(final_seq)
        
        # Get Latents
        z_t = transformer_out[:, -2, :] # [B, 768] Current State
        goal_token_repr = transformer_out[:, -1, :]
        policy_latent = self.policy_compressor(goal_token_repr)
        
        # 2. Get Hypotheses
        action_pred, action_scores = self.action_predictor(policy_latent) # [B, K, T, A], [B, K]
        
        # Process Actions (Cumsum, etc - same as forward)
        xy = torch.cumsum(action_pred[..., :2], dim=2)
        action_pred = torch.cat([xy, action_pred[..., 2:]], dim=-1)
        if self.learn_angle:
            action_pred[:, :, :, 2:] = F.normalize(action_pred[:, :, :, 2:].clone(), dim=-1)
            
        # 3. Imagination Loop (Batched)
        # We want to score each hypothesis k based on:
        # Score_k = (1-w) * Policy_Prob_k + w * Consistency(z_t, a_0^k, z_goal)
        
        # Extract first action a_0 for each hypothesis
        if self.learn_angle:
            # Action is [dx, dy, cos, sin], input to dynamics needs full params
            # But wait, action_pred is absolute position now? 
            # NO, MultiHypothesisHead outputs *deltas* natively, forward() does cumsum.
            # Dynamics "action" usually refers to the control input (delta).
            # The dynamics model was trained on the output of "actions" from dataset, which are usually waypoints.
            # In VINT/GNM, "action" passed to model is usually the ground truth waypoint sequence or the immediate next action.
            # Our code usage: `a_t = action`. In training, `action` is the label.
            # Label format: `actions` tensor from dataset.
            # It seems dataset returns full trajectory [T, A].
            # For dynamics, we probably just need the "command" implied by the first waypoint/action.
            # Let's assume we use the first step `action_pred[:, :, 0, :]` as the "action" to step dynamics.
            pass
        
        # To make batching easy: Flatten K into Batch
        # z_t: [B, 768] -> [B*K, 768]
        z_t_expanded = z_t.repeat_interleave(num_samples, dim=0)
        
        # Candidate Actions: [B, K, T, A] -> Take t=0 -> [B, K, A] -> [B*K, A]
        # We use the raw delta (first step) which represents the immediate intent.
        # Note: action_pred contains absolute positions after cumsum. 
        # But for t=0, absolute == delta (since start is 0,0).
        candidate_actions_t0 = action_pred[:, :, 0, :].reshape(B * num_samples, -1)
        
        # Embed Actions
        a_emb = self.action_embed(candidate_actions_t0) # [B*K, 128]
        
        # Predict Next State
        z_next_pred = self.dynamics(torch.cat([z_t_expanded, a_emb], dim=-1))
        z_next_pred = self.dyn_norm(z_next_pred)
        z_next_pred = F.normalize(z_next_pred, dim=-1) # [B*K, 768]
        
        # 4. Consistency Score with Goal
        # Maximize cosine sim(z_next, z_goal)
        # z_goal: [B, 768] -> [B*K, 768]
        z_goal_expanded = z_goal.repeat_interleave(num_samples, dim=0)
        
        goal_sim = (z_next_pred * z_goal_expanded).sum(dim=-1).view(B, num_samples) # [B, K]
        
        # Normalize scores to probability-like space if needed, or just combine raw logits
        # action_scores are logits. goal_sim is [-1, 1].
        # Let's simple linear combination? 
        # Better: Softmax policy scores first to get probability, then combine with normalized goal sim?
        # Or just treat goal_sim as a log-prob bonus?
        
        # Heuristic: Combined Score = Policy_Logit + w * (Goal_Sim * Scale)
        # Adding consistency as a bonus term.
        combined_scores = action_scores + (w_consistency * goal_sim * 10.0) 
        
        # Select Best
        best_idx = torch.argmax(combined_scores, dim=1) # [B]
        
        # Gather Best Trajectory
        # best_idx: [B] -> [B, 1, 1, 1]
        gather_idx = best_idx.view(B, 1, 1, 1).expand(-1, -1, action_pred.shape[2], action_pred.shape[3])
        best_traj = torch.gather(action_pred, 1, gather_idx).squeeze(1) # [B, T, A]
        
        return best_traj, combined_scores


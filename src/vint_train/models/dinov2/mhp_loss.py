import torch
import torch.nn as nn
import torch.nn.functional as F

class MHPLoss(nn.Module):
    def __init__(self, reg_loss_type='mse', smoothness_weight=0.1):
        """
        解耦位置与角度的解耦多假设预测损失。
        
        Args:
            reg_loss_type: 'mse', 'huber', 或 'smooth_l1'
            smoothness_weight: 轨迹平滑项的固定权重系数
        """
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.reg_loss_type = reg_loss_type
        
        if reg_loss_type == 'huber':
            self.reg_crit = nn.HuberLoss(reduction='none')
        elif reg_loss_type == 'smooth_l1':
            self.reg_crit = nn.SmoothL1Loss(beta=1.0, reduction='none')
        else:
            self.reg_crit = nn.MSELoss(reduction='none')
            
        self.cls_crit = nn.CrossEntropyLoss()

    def forward(self, pred_trajs, pred_scores, gt_traj):
        """
        Args:
            pred_trajs: [B, K, T, 4] (前2维是x,y; 后2维是sin,cos或角度特征)
            pred_scores: [B, K] 假设评分的logits
            gt_traj: [B, T, 4] 真实标签
            
        Returns:
            loss_dict: 包含各分量原始损失的字典，用于外部EMA加权
        """
        B, K, T, A = pred_trajs.shape
        gt_expand = gt_traj.unsqueeze(1) # [B, 1, T, 4]
        
        # 1. 寻找最优假设 (Winner-Takes-All)
        # 核心逻辑：仅基于【位置误差 (前2维)】来确定哪条轨迹是“赢家”
        diff_pos = pred_trajs[..., :2] - gt_expand[..., :2]
        ade_pos = torch.norm(diff_pos, dim=-1).mean(dim=-1) # [B, K]
        best_k_indices = torch.argmin(ade_pos, dim=1) 
        
        # 2. 提取最佳假设的完整 4 维信息
        # gather_idx 形状: [B, 1, T, 4]
        gather_idx = best_k_indices.view(B, 1, 1, 1).expand(-1, -1, T, A)
        best_pred_traj = torch.gather(pred_trajs, 1, gather_idx).squeeze(1) # [B, T, 4]
        
        # 3. 计算【位置回归损失】 (Pos Loss)
        pos_reg_raw = self.reg_crit(best_pred_traj[..., :2], gt_traj[..., :2]).mean()
        
        # 4. 计算【角度回归损失】 (Angle Loss)
        angle_reg_raw = self.reg_crit(best_pred_traj[..., 2:], gt_traj[..., 2:]).mean()
        
        # 5. 计算【平滑度损失】 (仅针对位置 x,y)
        # 2026.03.02 逻辑：惩罚相邻步长间的突变
        action_diff = best_pred_traj[:, 1:, :2] - best_pred_traj[:, :-1, :2]
        smoothness_loss = torch.norm(action_diff, dim=-1).mean()
        
        # 合并回归项 (位置项包含平滑度)
        loss_pos = pos_reg_raw + self.smoothness_weight * smoothness_loss
        loss_angle = angle_reg_raw
        
        # 6. 计算【分类损失】 (Score Loss)
        loss_score = self.cls_crit(pred_scores, best_k_indices)
        
        # 准确率统计
        acc = (torch.argmax(pred_scores, dim=1) == best_k_indices).float().mean()

        return {
            "loss_pos": loss_pos,
            "loss_angle": loss_angle,
            "loss_score": loss_score,
            "mhp_accuracy": acc.item()
        }
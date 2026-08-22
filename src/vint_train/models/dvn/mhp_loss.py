import torch
import torch.nn as nn
import torch.nn.functional as F

class MHPLoss(nn.Module):
    def __init__(
        self,
        reg_loss_type='smooth_l1',
        smoothness_weight=0.1,
        score_temp=1.0,
        angle_weight=1.0,
        assign_temp=1.0,
        diversity_weight=0.02,
        max_step=1.5,
        step_weight=0.05,
    ):
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.score_temp = score_temp
        self.angle_weight = angle_weight
        self.assign_temp = assign_temp
        self.diversity_weight = diversity_weight
        self.max_step = max_step
        self.step_weight = step_weight

        if reg_loss_type == 'huber':
            self.reg_crit = nn.HuberLoss(reduction='none')
        elif reg_loss_type == 'smooth_l1':
            self.reg_crit = nn.SmoothL1Loss(beta=1.0, reduction='none')
        else:
            self.reg_crit = nn.MSELoss(reduction='none')

    def soft_cross_entropy(self, logits, soft_targets):
        log_probs = F.log_softmax(logits, dim=-1)
        return -(soft_targets * log_probs).sum(dim=-1).mean()

    def forward(self, pred_trajs, pred_scores, gt_traj):
        B, K, T, A = pred_trajs.shape
        gt_expand = gt_traj.unsqueeze(1).expand(-1, K, -1, -1)

        pos_err = self.reg_crit(pred_trajs[..., :2], gt_expand[..., :2]).mean(dim=(-1, -2))  # [B,K]

        pred_dir = F.normalize(pred_trajs[..., 2:], dim=-1)
        gt_dir = F.normalize(gt_expand[..., 2:], dim=-1)
        cos_sim = (pred_dir * gt_dir).sum(dim=-1)  # [B,K,T]
        angle_err = (1 - cos_sim).mean(dim=-1)     # [B,K]

        total_err = pos_err + self.angle_weight * angle_err  # [B,K]

        # soft assignment for regression
        with torch.no_grad():
            assign_w = F.softmax(-total_err / self.assign_temp, dim=1)  # [B,K]

        loss_pos = (assign_w * pos_err).sum(dim=1).mean()
        loss_angle = (assign_w * angle_err).sum(dim=1).mean()

        # score supervision
        soft_targets = F.softmax(-total_err / self.score_temp, dim=1)
        loss_score = self.soft_cross_entropy(pred_scores, soft_targets)

        # smoothness on all hypotheses (weighted)
        action_diff = pred_trajs[:, :, 1:, :2] - pred_trajs[:, :, :-1, :2]  # [B,K,T-1,2]
        smooth_per_k = torch.norm(action_diff, dim=-1).mean(dim=-1)         # [B,K]
        smoothness_loss = (assign_w * smooth_per_k).sum(dim=1).mean()

        # step-length constraint
        step_norm = torch.norm(pred_trajs[..., :2], dim=-1)   # [B,K,T]
        loss_step = F.relu(step_norm - self.max_step).mean()

        # pred_trajs are cumulative waypoints at this stage, so the endpoint is
        # the final waypoint directly. This also keeps K=1 ablations finite.
        if K > 1 and self.diversity_weight > 0:
            end_pts = pred_trajs[..., -1, :2]  # [B,K,2]
            pdist = torch.cdist(end_pts, end_pts)  # [B,K,K]
            eye = torch.eye(K, device=pdist.device).unsqueeze(0)
            div_loss = F.relu(0.8 - pdist) * (1 - eye)
            div_loss = div_loss.sum() / (B * K * (K - 1))
        else:
            div_loss = pred_trajs.new_zeros(())

        hard_pred = torch.argmax(pred_scores, dim=1)
        hard_best = torch.argmin(total_err, dim=1)
        acc = (hard_pred == hard_best).float().mean()
        loss_pos = loss_pos + self.smoothness_weight * smoothness_loss + self.step_weight * loss_step
        mhp_loss = loss_pos + loss_angle + loss_score + self.diversity_weight * div_loss
        return {
            "loss_pos": loss_pos,
            "loss_angle": loss_angle,
            "loss_score": loss_score,
            "loss_div": self.diversity_weight * div_loss,
            "mhp_loss": mhp_loss,
            "mhp_accuracy": acc.item(),
        }

import torch
from .safety_utils import project_and_sample
from .safety_loss import DifferentiableCollisionLoss

def select_safe_action(trajs, scores, depth_img, intrinsics=None, lambda_safety=1.0):
    """
    Selects the best trajectory by penalizing collision candidates.
    Score = Model_Score - lambda * Collision_Cost
    """
    B, K, T, _ = trajs.shape
    
    if intrinsics is None:
        # Create dummy instance to get defaults
        # Ideally this should be passed in or stored in config
        dummy_loss = DifferentiableCollisionLoss()
        intrinsics = dummy_loss.get_default_intrinsics(B, trajs.device)
        
    with torch.no_grad():
        collision_cost = project_and_sample(trajs, depth_img, intrinsics) # [B, K]
        
    # Scores are logits (e.g. from Softmax or raw). Assuming raw.
    # Higher score = better. Lower cost = better.
    # Final Metric = Score - lambda * Cost
    
    final_score = scores - (lambda_safety * collision_cost)
    
    best_idx = torch.argmax(final_score, dim=1) # [B]
    
    # Gather best traj
    # trajs: [B, K, T, 2]
    # best_idx: [B] -> need [B, 1, 1, 1] expansion or fancy indexing
    batch_indices = torch.arange(B, device=trajs.device)
    best_traj = trajs[batch_indices, best_idx] # [B, T, 2]
    
    return best_traj

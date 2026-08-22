### dvn 1.0
目前的效果还可以，但是存在一些问题：
引入深度图增加碰撞损失，但是需要相机内外参，目前设置了一个通用的内外参，但不同数据集的内外参并不一致，而投影不对可能导致模型出错

### dvn2.0
去掉碰撞损失，引入下面的思路：
1. 训练阶段不进行碰撞损失预测，而推理阶段，因为有真实的相机内外参，因此增加如果有深度信息，则预测碰撞损失，当并对score进行修正：
```python
@torch.no_grad()
def inference_with_safety(model, obs, goal, depth, intrinsics, camera_height):
    dist_pred, action_pred, action_scores, _, _ = model(obs, goal)
    
    collision_costs = project_and_sample(
        action_pred, depth, intrinsics, 
        camera_height=camera_height, 
        safety_radius=0.4
    )
    lambda_safety = 5.0 
    final_scores = action_scores - lambda_safety * collision_costs
    
    best_idx = torch.argmax(final_scores, dim=1)
    
    final_trajectory = action_pred[0, best_idx]
    return final_trajectory
```

2. 针对动力学损失的优化
    1. 动态置信度加权：动力学模型预测的损失，动态调整上一个的避障权重
    2. 潜在空间多次预测，不仅预测一次，类似nomad的扩散
    3. 测试时在线自适应
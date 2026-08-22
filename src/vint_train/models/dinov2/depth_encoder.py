import torch
import torch.nn as nn
import timm

class DepthEncoder(nn.Module):
    def __init__(self, output_dim=768, backbone_name='resnet18'):
        super().__init__()
        # Load backbone with 1 channel input
        # num_classes=0 keeps the pooling layer usually, but we will use forward_features
        self.backbone = timm.create_model(backbone_name, pretrained=True, in_chans=1, num_classes=0)
        self.backbone = replace_bn_with_gn(self.backbone)
        
        # Determine feature dim
        with torch.no_grad():
            dummy = torch.randn(1, 1, 224, 224)
            # forward_features returns [B, C, H, W] for ResNet
            feat = self.backbone.forward_features(dummy)
            self.input_dim = feat.shape[1]
            
        # Projector: [C, 1, 1] -> [output_dim]
        # We use a 1x1 conv to project channel dim while keeping spatial dims
        assert output_dim % 8 == 0
        self.projector = nn.Sequential(
            nn.Conv2d(self.input_dim, output_dim, kernel_size=1),
            nn.GroupNorm(8, output_dim), # Better than LayerNorm for spatial
            nn.ReLU()
        )

    def forward(self, x):
        """
        Input: [B, 1, H, W]
        Output: [B, N, output_dim] where N is H'*W'
        """
        features = self.backbone.forward_features(x) # [B, C, h, w]
        projected = self.projector(features)         # [B, out_dim, h, w]
        
        # Flatten spatial dimensions to tokens
        # [B, out_dim, h, w] -> [B, out_dim, h*w] -> [B, h*w, out_dim]
        B, C, H, W = projected.shape
        tokens = projected.flatten(2).transpose(1, 2)
        
        return tokens

# Utils for Group Norm
def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups = max(1, x.num_features // features_per_group),
            num_channels=x.num_features)
    )
    return root_module


def replace_submodules(
        root_module: nn.Module,
        predicate: callable,
        func: callable) -> nn.Module:
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    return root_module

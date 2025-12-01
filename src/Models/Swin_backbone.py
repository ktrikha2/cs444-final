import timm
import torch.nn as nn
from timm.layers import PatchEmbed

class SwinBackbone(nn.Module):
    """Create a feature-extractor from timm's Swin model usable by torchvision's FasterRCNN.

    We extract features from the final feature map and provide an out_channels attribute.
    This is a minimal adapter — for better performance you'd extract a feature pyramid.
    """

    def __init__(self, variant='swin_tiny_patch4_window7_224', pretrained=True):
        super().__init__()
        # create a swin model with global pooling disabled so we can get the spatial map
        self.net = timm.create_model(
            variant, 
            pretrained=pretrained, 
            features_only=True
            #strict_img_size=False, 
            #img_size=768
        )        # features_only returns a list of stage feature maps; choose the last stage
        for m in self.net.modules():
            if isinstance(m, PatchEmbed):
                m.strict_img_size = False
        feat_channels = self.net.feature_info.channels()[-1]
        self.out_channels = feat_channels

    def forward(self, x):
        # features_only -> list of multi-scale feature maps, pick the last one
        feats = self.net(x)
        feat = feats[-1]
        
        # DEBUG: Print feature information
        print(f"\n=== Swin Backbone Debug ===")
        print(f"Input x shape: {x.shape}")
        print(f"Number of feature maps returned: {len(feats)}")
        for i, f in enumerate(feats):
            print(f"  Feature map {i} shape: {f.shape}")
        print(f"Selected feature (feats[-1]) shape: {feat.shape}")
        print(f"Feature dtype: {feat.dtype}")
        
        # CRITICAL: timm's Swin returns features in [B, H, W, C] format
        # but torchvision FasterRCNN expects [B, C, H, W]
        # Check if we need to permute
        if feat.dim() == 4 and feat.shape[1] < feat.shape[3]:
            print(f"Permuting from [B, H, W, C] to [B, C, H, W]")
            feat = feat.permute(0, 3, 1, 2).contiguous()
            print(f"After permute shape: {feat.shape}")
        else:
            print(f"No permute needed, shape already correct")

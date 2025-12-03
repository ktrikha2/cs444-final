# src/Models/SwinBackbone.py
import torch
from torch import nn
import timm
from timm.layers import PatchEmbed


class SwinBackbone(nn.Module):
    #Thin wrapper around a timm Swin model in features_only mode. Returns only the last stage feature map.
    def __init__(self, cfg):
        super().__init__()

        swin_cfg = cfg["model"]
        swin_variant = swin_cfg.get("swin_variant", "swin_tiny_patch4_window7_224")
        swin_pretrained = swin_cfg.get("pretrained", True)

        # swin feature extractor
        self.swin = timm.create_model(
            swin_variant,
            pretrained=swin_pretrained,
            features_only=True,
            out_indices=(3,),  # last stage only
        )
        for m in self.swin.modules():
            if isinstance(m, PatchEmbed):
                m.strict_img_size = False
        # Number of channels in the last stage (C_out)
        self.out_channels = self.swin.feature_info.channels()[-1]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, 3, H, W]
        returns: feature map [B, C_out, H_s, W_s]
        """
        features = self.swin(images)   # list with 1 element because out_indices=(3,)
        feat = features[0]
        return feat


def build_swin_backbone(cfg) -> SwinBackbone:
    return SwinBackbone(cfg)

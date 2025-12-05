# src/Models/SwinBackbone.py
import torch
from torch import nn
import timm
from timm.layers import PatchEmbed
import torch.nn.functional as F


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
        for p in self.swin.parameters():
            p.requires_grad = False
        self.out_channels = self.swin.feature_info.channels()[-1]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, 3, H, W]
        returns: feature map [B, C_out, H_s, W_s]
        """

        #NEED TO ADD PADDING TO WORK WITH SWIN PRETRAINED
        B, C, H, W = images.shape
        pad_h = (224 - H % 224) % 224
        pad_w = (224 - W % 224) % 224

        if pad_h > 0 or pad_w > 0:
            images = F.pad(images, (0, pad_w, 0, pad_h))

        features = self.swin(images)   # list with 1 element because out_indices=(3,)
        feat = features[0]

        feat = feat.permute(0, 3, 1, 2).contiguous() #changing from BCHW to BHWC in the tensor

        #print("BACKBONE OUT SHAPE:", feat.shape, flush=True)
        
        return feat


def build_swin_backbone(cfg) -> SwinBackbone:
    return SwinBackbone(cfg)

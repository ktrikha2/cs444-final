# src/Models/SwinDetr.py
import torch
from torch import nn

from .SwinBackbone import build_swin_backbone
from .SwinDetrHead import SwinDetrHead


class SwinDETR(nn.Module):
    """
    Full Swin-DETR model:
      images -> Swin backbone -> DETR head -> pred_logits, pred_boxes
    """
    def __init__(self, cfg):
        super().__init__()

        # Backbone
        self.backbone = build_swin_backbone(cfg)
        backbone_out_channels = self.backbone.out_channels

        # DETR head
        self.detr_head = SwinDetrHead(cfg, backbone_out_channels)

    def forward(self, images: torch.Tensor):
        """
        images: [B, 3, H, W]
        returns dict with 'pred_logits' and 'pred_boxes'
        """
        feat = self.backbone(images)   # [B, C_out, H_s, W_s]
        outputs = self.detr_head(feat)
        print("BACKBONE OUT SHAPE:", feat.shape)
        print("BACKBONE stats mean/std:", feat.mean().item(), feat.std().item(), flush=True)
        return outputs


def build_swin_detr(cfg) -> SwinDETR:
    return SwinDETR(cfg)

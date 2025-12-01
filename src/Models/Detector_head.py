import torch
from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from src.Models.Swin_backbone import SwinBackbone

def make_model(cfg):
    backbone = SwinBackbone(
        variant=cfg['model']['swin_variant'],
        pretrained=cfg['model']['pretrained']
    )

    # torchvision FasterRCNN expects a backbone that returns an OrderedDict of
    # feature maps with an attribute "out_channels". We'll wrap the single-stage
    # feature map in a small adapter.
    class BackboneWrapper(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.body = backbone
            self.out_channels = backbone.out_channels

        def forward(self, x):
            # return dict of one feature map with key "0"
            feat = self.body(x)
            return {'0': feat}

    wrapped = BackboneWrapper(backbone)

    # anchor sizes and aspect ratios can be tuned
    anchor_generator = AnchorGenerator(
        sizes=tuple(cfg['model'].get('rpn_anchor_sizes', [[32, 64, 128, 256, 512]])),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # box predictor is left default; num_classes is a placeholder here
    model = FasterRCNN(
        wrapped,
        num_classes=2,
        rpn_anchor_generator=anchor_generator,
        min_size=768,
        max_size=768
    )

    return model

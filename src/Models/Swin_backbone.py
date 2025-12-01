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
        print(feats)
        return feats[-1]

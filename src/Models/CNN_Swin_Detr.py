import torch
import torch.nn as nn
import math
from .CNN_Swin import SwinDETRBackbone

class Neck(nn.Module):
    def __init__(self, in_dim=1024, out_dim=256, num_encoder_layers=6, nhead=8, dim_feedforward=512):
        super().__init__()
        # 1x1 conv to reduce channels
        self.conv1x1 = nn.Linear(in_dim, out_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(out_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=out_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    def forward(self, x):
        # x: [B, N, C_in]
        x = self.conv1x1(x)               # [B, N, out_dim]
        x = x + self.pos_encoding(x)      # add positional encoding
        x = self.encoder(x)               # [B, N, out_dim]
        return x

class PositionalEncoding(nn.Module):
    """Sine-cosine positional encoding"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, N, d_model]
        return self.pe[:, :x.size(1), :]

class DETRDecoder(nn.Module):
    def __init__(self, d_model=256, num_queries=100, nhead=8, num_layers=6, dim_feedforward=512):
        super().__init__()
        # Learnable object queries
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, d_model)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, encoder_output):
        # encoder_output: [B, N, d_model]
        B = encoder_output.size(0)
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, num_queries, d_model]
        out = self.decoder(tgt=queries, memory=encoder_output)          # [B, num_queries, d_model]
        return out

class PredictionHead(nn.Module):
    def __init__(self, d_model=256, hidden_dim=256, num_classes=80):
        super().__init__()
        # MLP for bounding box regression
        self.bbox_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)   # [cx, cy, w, h], normalized
        )
        # Linear layer for class prediction
        self.class_embed = nn.Linear(d_model, num_classes + 1)  # +1 for "no object" class

    def forward(self, x):
        # x: [B, num_queries, d_model]
        boxes = self.bbox_mlp(x).sigmoid()   # normalized coordinates
        classes = self.class_embed(x)        # logits for softmax later
        return boxes, classes

class SwinDETR(nn.Module):
    def __init__(self, backbone, num_queries=100, num_classes=80):
        super().__init__()
        self.backbone = backbone
        self.neck = Neck(in_dim=1024, out_dim=256)
        self.decoder = DETRDecoder(d_model=256, num_queries=num_queries)
        self.head = PredictionHead(d_model=256, hidden_dim=256, num_classes=num_classes)

    def forward(self, x):
        # Backbone
        features, H, W = self.backbone(x)  # [B, N, C_backbone]
        # Neck
        encoder_output = self.neck(features)  # [B, N, d_model]
        # Decoder
        decoder_output = self.decoder(encoder_output)  # [B, num_queries, d_model]
        # Prediction Head
        boxes, classes = self.head(decoder_output)
        outputs = {
        "pred_boxes": boxes,       # [B, num_queries, 4]
        "pred_logits": classes     # [B, num_queries, num_classes+1]
        }
        return outputs

def build_swin_detr(cfg):
    backbone = SwinDETRBackbone(
        embed_dim=cfg["model"]["embed_dim"],
        num_heads=cfg["model"]["num_heads"],
        num_blocks=cfg["model"]["num_blocks"],
        window_size=cfg["model"]["window_size"]
    )

    model = SwinDETR(
        backbone=backbone,
        num_queries=cfg["model"]["decoder_queries"],
        num_classes=cfg["model"]["num_classes"]
    )
    return model

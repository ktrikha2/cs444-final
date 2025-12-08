import torch
import torch.nn as nn
import math
from .CNN_Swin import SwinDETRBackbone

class Neck(nn.Module):
    def __init__(self, in_dim=1024, out_dim=256, num_encoder_layers=6, nhead=8, dim_feedforward=512):
        super().__init__()
        # 1x1 conv to reduce channels
        self.conv1x1 = nn.Conv2d(in_dim, out_dim, kernel_size=1) #changing from linear to conv 2d
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(out_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=out_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    def forward(self, x):
        # x: [B, N, C_in] NOW IT IS [B, C_IN, H, W]
        #print("Neck input:", x.mean().item(), x.std().item()) 
        B,C,H,W = x.shape
        x = self.conv1x1(x)               # [B, N, out_dim]

        device = x.device
        pe_2d = self.pos_encoding(H, W, device)
        pos = pe_2d.unsqueeze(0).repeat(B,1,1)

        x = x.flatten(2).transpose(1,2)
        #print("After 1x1 conv:", x.mean().item(), x.std().item())
        #pos = self.pos_encoding(x)
        #print("PosEnc stats:", pos.mean().item(), pos.std().item())
        x = x + pos     # add positional encoding
        #print("After adding pos:", x.mean().item(), x.std().item())

        x = self.encoder(x)               # [B, N, out_dim]
        #print("Neck encoder out:", x.mean().item(), x.std().item())

        return x

class PositionalEncoding(nn.Module):
    """2D Sine-cosine positional encoding (DETR-style), dynamic size"""
    def __init__(self, d_model, temperature=10000):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
        
        # We need d_model // 2 for X and Y combined
        d_quarter = d_model // 4
        
        # Precompute the division term once
        div_term = torch.exp(torch.arange(0, d_quarter, 
                                          dtype=torch.float32) * (-math.log(temperature) / d_quarter))
        self.register_buffer('div_term', div_term)

    def forward(self, H, W, device):
        # Create coordinates
        y_embed = torch.arange(H, dtype=torch.float32, device=device).unsqueeze(1)
        x_embed = torch.arange(W, dtype=torch.float32, device=device).unsqueeze(1)
        
        # pe_x, pe_y: [*, d_quarter]
        pe_x = x_embed * self.div_term
        pe_y = y_embed * self.div_term

        # pe_x: [W, d_model/2], pe_y: [H, d_model/2] (by stacking sin/cos)
        pe_x = torch.stack([pe_x.sin(), pe_x.cos()], dim=-1).flatten(1) 
        pe_y = torch.stack([pe_y.sin(), pe_y.cos()], dim=-1).flatten(1) 
        
        # Combine to 2D grid
        pe_x_2d = pe_x.unsqueeze(0).repeat(H, 1, 1)
        pe_y_2d = pe_y.unsqueeze(1).repeat(1, W, 1)
        
        # Final PE: [H, W, d_model]
        pe = torch.cat([pe_y_2d, pe_x_2d], dim=-1)

        # Flatten: [H*W, d_model]
        return pe.flatten(0, 1)

class DETRDecoder(nn.Module):
    def __init__(self, d_model=256, num_queries=100, nhead=8, num_layers=6, dim_feedforward=512):
        super().__init__()
        # Learnable object queries
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        self.tgt_embed = nn.Embedding(num_queries, d_model)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, encoder_output):
        # encoder_output: [B, N, d_model]
        B = encoder_output.size(0)
        #print("Decoder input:", encoder_output.mean().item(), encoder_output.std().item())
        tgt = self.tgt_embed.weight.unsqueeze(0).repeat(B, 1, 1)

        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, num_queries, d_model]
        tgt_with_pos = tgt + queries
        #with torch.no_grad():
            #print("query_embed std:", self.query_embed.weight.std(dim=0).mean().item())
            #print("tgt_embed std:", self.tgt_embed.weight.std(dim=0).mean().item())
            #print("tgt (content query) mean/std:", tgt.mean().item(), tgt.std().item())
            #print("queries (pos query) mean/std:", queries.mean().item(), queries.std().item())
        out = self.decoder(tgt=tgt_with_pos, memory=encoder_output)          # [B, num_queries, d_model]
        #print("Decoder output:", out.mean().item(), out.std().item())
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
        nn.init.constant_(self.bbox_mlp[-1].bias.data, 0)
        #nn.init.constant_(self.bbox_mlp[-1].weight.data, 0)
        nn.init.xavier_uniform_(self.bbox_mlp[-1].weight.data)
        print("INIT CHECK:", 
            self.bbox_mlp[-1].weight.std().item(), 
            self.bbox_mlp[-1].bias.std().item())

        # Linear layer for class prediction
        self.class_embed = nn.Linear(d_model, num_classes + 1)  # +1 for "no object" class
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.class_embed.bias.data[num_classes], bias_value)

    def forward(self, x):
        # x: [B, num_queries, d_model]
        #print("Head input:", x.mean().item(), x.std().item())

        boxes = self.bbox_mlp(x).sigmoid()   # normalized coordinates
        #print("Raw bbox mlp:", boxes.mean().item(), boxes.std().item())
        #print("bbox layer final weight std:", self.bbox_mlp[-1].weight.std().item())
        #print("bbox layer final bias std:", self.bbox_mlp[-1].bias.std().item())
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
        features = self.backbone(x)  # [B, C_backbone, h, w]
        #print("Backbone features:", features.mean().item(), features.std().item())

        # Neck
        encoder_output = self.neck(features)  # Will be [B, N, d_model]
        # Decoder
        decoder_output = self.decoder(encoder_output)  # [B, num_queries, d_model]
        # Prediction Head
        with torch.no_grad():
            if encoder_output.size(0) > 1: # Only if batch size > 1
                diff = (encoder_output[0] - encoder_output[1]).abs().mean().item()
                #print("encoder_output image difference:", diff)
            # std_across_queries is the std for each dimension, calculated across all queries.
            std_across_queries = decoder_output[0].std(dim=0) 
            #print("decoder output per-query std (mean):", std_across_queries.mean().item())
            #print("decoder output per-query std (min):", std_across_queries.min().item())
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

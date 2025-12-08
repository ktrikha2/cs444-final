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
        #print("Neck encoder out:", x.mean().item(), x.std().item())\
        #print("Encoder output mean/std:", x.mean().item(), x.std().item())
        #print("Encoder output spatial diff:",
            #(x[:,1:,:] - x[:,:-1,:]).abs().mean().item())

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
        self.d_model = d_model
        #self.tgt_embed = nn.Embedding(num_queries, d_model)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, encoder_output):
        # encoder_output: [B, N, d_model]
        B = encoder_output.size(0)
        #print("Decoder input:", encoder_output.mean().item(), encoder_output.std().item())
        #tgt = self.tgt_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        #tgt = torch.zeros(B, self.num_queries, self.d_model, device=encoder_output.device)
        print("Query embed std:", self.query_embed.weight.std().item())
        print("Query embed mean:", self.query_embed.weight.mean().item())

        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, num_queries, d_model]
        #tgt_with_pos = tgt + queries
        #with torch.no_grad():
            #print("query_embed std:", self.query_embed.weight.std(dim=0).mean().item())
            #print("tgt_embed std:", self.tgt_embed.weight.std(dim=0).mean().item())
            #print("tgt (content query) mean/std:", tgt.mean().item(), tgt.std().item())
            #print("queries (pos query) mean/std:", queries.mean().item(), queries.std().item())
        out = self.decoder(tgt=queries, memory=encoder_output)        # [B, num_queries, d_model]
        #print("Decoder output:", out.mean().item(), out.std().item())
        #print("Decoder output std:", out.std().item())
        #print("Decoder output first-row:", out[0, :5].detach().cpu())

        return out




class MLP(nn.Module):
    """DETR-style 3-layer MLP for bbox regression."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers - 1):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)

        # DETR uses Xavier init for all linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layers(x)


class PredictionHead(nn.Module):
    def __init__(self, d_model=256, hidden_dim=256, num_classes=80):

        super().__init__()

        # DETR bbox MLP (3 layers): d_model → hidden_dim → hidden_dim → 4
        self.bbox_mlp = MLP(d_model, hidden_dim, 4, num_layers=3)

        # Classification head
        self.class_embed = nn.Linear(d_model, num_classes + 1)

        # Initialize no-object bias like DETR
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.class_embed.bias.data[-1], bias_value)

    def forward(self, x):
        """
        x: [B, num_queries, d_model]
        """


        if self.training and x.shape[1] > 0:
            with torch.no_grad():
                print("[HEAD INPUT] first row first 8 dims:",
                      x[0, 0, :8].detach().cpu())
                print("[HEAD INPUT] per-dim std across queries (first 8 dims):",
                      x[0, :, :8].std(dim=0).detach().cpu())
                print("[HEAD INPUT] mean/std over all queries:",
                      x.mean().item(), x.std().item())


        raw = self.bbox_mlp(x)
        if not self.training:
            with torch.no_grad():
                print("[RAW BBOX] per-dim std:", raw[0].std(dim=0).cpu().tolist())
                print("[RAW BBOX] first 5:", raw[0, :5].cpu())
        if self.training and raw.shape[1] > 0:
            with torch.no_grad():
                per_dim_std = raw[0].std(dim=0)
                print("[RAW BBOX] per-dim std:", per_dim_std.cpu().tolist())
                print("[RAW BBOX] first 5 rows:", raw[0, :5].detach().cpu())

        # Sigmoid → normalized cxcywh [0,1]
        boxes = raw.sigmoid()

        # classification logits
        classes = self.class_embed(x)

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
        print("[BACKBONE] forward called. Feature std:", x.std().item())

        features = self.backbone(x)  # [B, C_backbone, h, w]
        print("[SWIN DETR] backbone output std:", features.std().item())

        # Neck
        encoder_output = self.neck(features)  # Will be [B, N, d_model]
        print("\n[DEBUG] MEMORY INPUT SHAPE:", encoder_output.shape)
        print("[DEBUG] MEMORY FIRST TOKEN FIRST 8 VALUES:", encoder_output[0, 0, :8])
        print("[DEBUG] MEMORY MEAN/STD:", encoder_output.mean().item(), encoder_output.std().item())

        # Decoder
        decoder_output = self.decoder(encoder_output)  # [B, num_queries, d_model]
        # Prediction Head
        if self.training and torch.rand(1).item() < 0.01:
            std_queries = decoder_output.std(dim=1).mean().item()
            print(f"\n[DEBUG] Decoder Output Variance (Std across queries): {std_queries:.6f}")
            if std_queries < 0.001:
                print("DECODER COLLAPSED (Inputs identical)")
            else:
                print("DECODER HEALTHY (Inputs diverse)")
        if not self.training:
            with torch.no_grad():
                print("[EVAL] decoder_output mean/std:",
                      decoder_output.mean().item(),
                      decoder_output.std().item())

                print("[EVAL] decoder_output first-row first-8 dims:",
                      decoder_output[0, 0, :8])
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

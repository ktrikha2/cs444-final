# src/Models/SwinDetrHead.py
import torch
from torch import nn


class PositionalEncoding2D(nn.Module):
    """
    2D sine-cosine positional encoding.
    Shape should be [B, C, H, W] where C == hidden_dim.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be even for 2D positional encoding")
        self.hidden_dim = hidden_dim

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device

        # We split hidden_dim as: half for Y, half for X
        dim_half = self.hidden_dim // 2

        dim_t = torch.arange(dim_half, dtype=torch.float32, device=device)
        dim_t = 10000 ** (2 * (dim_t // 2) / dim_half)

        y_embed = torch.arange(H, dtype=torch.float32, device=device).unsqueeze(1)
        x_embed = torch.arange(W, dtype=torch.float32, device=device).unsqueeze(1)

        pos_y = y_embed / dim_t          # [H, dim_half]
        pos_x = x_embed / dim_t          # [W, dim_half]

        pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=2).flatten(1)
        pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=2).flatten(1)

        # Now each is exactly dim_half
        pos_y = pos_y[:, :dim_half]
        pos_x = pos_x[:, :dim_half]

        pos = torch.cat(
            (
                pos_y[:, None, :].repeat(1, W, 1),   # [H, W, dim_half]
                pos_x[None, :, :].repeat(H, 1, 1),   # [H, W, dim_half]
            ),
            dim=2,
        )  # [H, W, hidden_dim]

        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1)
        return pos


class MLP(nn.Module):
    """Simple 3-layer MLP for bbox prediction Not full DETR yet."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        super().__init__()
        layers = []
        d_in = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(d_in, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            d_in = hidden_dim
        layers.append(nn.Linear(d_in, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SwinDetrHead(nn.Module):
    """
    Input:  feat [B, C_in, H, W]
    Output: dict with:
      - pred_logits: [B, num_queries, num_classes+1]
      - pred_boxes:  [B, num_queries, 4] (normalized cx,cy,w,h in [0,1])
    """
    def __init__(self, cfg, backbone_out_channels: int):
        super().__init__()

        model_cfg = cfg["model"]

        self.hidden_dim = model_cfg.get("hidden_dim", 256)
        num_classes = model_cfg.get("num_classes", 2)
        self.num_queries = model_cfg.get("num_queries", 100)

        # 1channel reduction 1x1 conv 
        self.channel_reduction = nn.Conv2d(
            backbone_out_channels, self.hidden_dim, kernel_size=1
        )
        self.input_norm = torch.nn.LayerNorm(self.hidden_dim)

        # pos encoding
        self.pos_enc = PositionalEncoding2D(self.hidden_dim)

        # Transformer encoder/decoder
        nheads = model_cfg.get("nheads", 8)
        enc_layers = model_cfg.get("enc_layers", 6)
        dec_layers = model_cfg.get("dec_layers", 6)
        dim_feedforward = model_cfg.get("dim_feedforward", 2048)
        dropout = model_cfg.get("dropout", 0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=False,  # [S, B, C]
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)

        # Object queries
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)

        # Prediction heads
        # +1 for no object / background class
        self.class_embed = nn.Linear(self.hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, num_layers=3)

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize roughly like DETR
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.constant_(self.bbox_embed.net[-1].bias, 0.0)
        nn.init.constant_(self.class_embed.bias, 0.0)

    def forward(self, feat: torch.Tensor):
        """
        feat: [B, C_in, H, W] from SwinBackbone
        returns: dict(pred_logits, pred_boxes)
        """
        b, _, h, w = feat.shape

        # Channel reduction
        feat = self.channel_reduction(feat)  # [B, hidden_dim, H, W]

        #pos encoding
        pos = self.pos_enc(feat)            # [B, hidden_dim, H, W]
        #print("POS stats:", pos.mean().item(), pos.std().item())

        if feat.shape[1] != pos.shape[1]:
            print("FEAT:", feat.shape, "POS:", pos.shape, flush=True)

        feat = feat + pos

        B,C,H,W = feat.shape
        feat = feat.permute(0,2,3,1)
        feat = self.input_norm(feat)
        feat = feat.permute(0,3,1,2).contiguous()

        #  Flatten spatial dims for Transformer to  [S, B, C]
        feat_flat = feat.flatten(2).permute(2, 0, 1).contiguous()  # [H*W, B, C]
        #print("HEAD INPUT stats:", feat_flat.mean().item(), feat_flat.std().item(), flush=True)

        # Encoder
        memory = self.encoder(feat_flat)  # [S, B, C]
        #print("MEMORY stats:", memory.mean().item(), memory.std().item(), flush=True)


        # Object queries as decoder targets
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, b, 1)  # [num_queries, B, C]
        tgt = torch.zeros_like(query_embed)

        # Decoder
        hs = self.decoder(tgt=tgt + query_embed, memory=memory)  # [num_queries, B, C]
        hs = hs.permute(1, 0, 2)                   # [B, num_queries, C]
        #print("DECODER HS stats:", hs.mean().item(), hs.std().item(), flush=True)
        # Heads
        pred_logits = self.class_embed(hs)          # [B, num_queries, num_classes+1]
        pred_boxes  = self.bbox_embed(hs).sigmoid() # [B, num_queries, 4] in [0,1]

        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}

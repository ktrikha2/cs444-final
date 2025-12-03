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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        device = x.device

        dim_t = torch.arange(self.hidden_dim // 2, dtype=torch.float32, device=device)
        dim_t = 10000 ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_y = torch.arange(h, dtype=torch.float32, device=device).unsqueeze(1)
        pos_x = torch.arange(w, dtype=torch.float32, device=device).unsqueeze(1)

        pos_y = pos_y / dim_t.unsqueeze(0)  # [H, D/2]
        pos_x = pos_x / dim_t.unsqueeze(0)  # [W, D/2]

        pos_y = torch.stack([torch.sin(pos_y), torch.cos(pos_y)], dim=2).flatten(1)
        pos_x = torch.stack([torch.sin(pos_x), torch.cos(pos_x)], dim=2).flatten(1)

        # build full grid
        pos = torch.zeros((h, w, self.hidden_dim), device=device)
        for i in range(h):
            for j in range(w):
                pos[i, j, : self.hidden_dim // 2] = pos_y[i]
                pos[i, j, self.hidden_dim // 2 :] = pos_x[j]

        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(b, 1, 1, 1)  # [B, C, H, W]
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
        feat = feat + pos

        #  Flatten spatial dims for Transformer to  [S, B, C]
        feat_flat = feat.flatten(2).permute(2, 0, 1).contiguous()  # [H*W, B, C]

        # Encoder
        memory = self.encoder(feat_flat)  # [S, B, C]

        # Object queries as decoder targets
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, b, 1)  # [num_queries, B, C]
        tgt = torch.zeros_like(query_embed)

        # Decoder
        hs = self.decoder(tgt=tgt, memory=memory)  # [num_queries, B, C]
        hs = hs.permute(1, 0, 2)                   # [B, num_queries, C]

        # Heads
        pred_logits = self.class_embed(hs)          # [B, num_queries, num_classes+1]
        pred_boxes  = self.bbox_embed(hs).sigmoid() # [B, num_queries, 4] in [0,1]

        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}

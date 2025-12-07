import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0,0,0,pad_w,0,pad_h)) 
    H_pad, W_pad = x.shape[1], x.shape[2]
    x = x.view(B, H_pad // window_size, window_size, W_pad // window_size, window_size, C)
    windows = x.permute(0,1,3,2,4,5).contiguous().view(-1, window_size, window_size, C)
    return windows, H_pad, W_pad

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        H, W: original padded height and width
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0,1,3,2,4,5).contiguous().view(B, H, W, -1)
    return x

def compute_attn_mask(H: int, W: int, window_size: int, shift_size: int, device: torch.device) -> torch.Tensor:
    """
    Builds attention mask for shifted windows as in the Swin paper.
    Returns mask with shape (num_windows, window_size*window_size, window_size*window_size)
    """
    if shift_size == 0:
        return None

    img_mask = torch.zeros((1, H, W, 1), device=device)  # 1 H W 1
    cnt = 0
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    # partition windows
    mask_windows = window_partition(img_mask, window_size)  # (num_windows, ws, ws, 1)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask  # shape (num_windows, ws*ws, ws*ws)


# --- CNN Downsampling ---
class DownsampleCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=128, kernel_size=4, stride=4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        #self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)       # [B, 128, H/4, W/4]
        #x = self.relu(x)
        return x

# --- Window Attention ---
class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

        #Define relative position bias table (learnable parameter)
        table_size = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(table_size, num_heads)
        ) 
        #coords for relative positive index 
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Ws, Ws
        coords_flatten = torch.flatten(coords, 1)  # 2, Ws*Ws

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        # relative_coords: (Ws*Ws, Ws*Ws, 2)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() 
        relative_coords[:, :, 0] += self.window_size - 1  # shift to be positive: [0, 2*Ws-2]
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1 #mapping back to 1 d index
        relative_position_index = relative_coords.sum(-1) 
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)


    def forward(self, x, mask: torch.Tensor = None):
        """
        x: (B_*N_windows, N, C)
        mask: (num_windows, N, N) optional
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )  # Ws*Ws, Ws*Ws, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Ws*Ws, Ws*Ws
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            B = B_ // nW
            attn = attn.view(B, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(2)
            attn = attn.view(B_, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1,2).reshape(B_, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

# --- Swin Transformer Block ---
class SwinTransformerBlock(nn.Module):
    # def __init__(self, dim, num_heads, window_size=7, shift_size = 0, mlp_ratio=4.0):
    #     super().__init__()
    #     self.norm1 = nn.LayerNorm(dim)
    #     self.attn = WindowAttention(dim, num_heads, window_size)
    #     self.norm2 = nn.LayerNorm(dim)
    #     self.mlp = nn.Sequential(
    #         nn.Linear(dim, int(dim * mlp_ratio)),
    #         nn.GELU(),
    #         nn.Linear(int(dim * mlp_ratio), dim)
    #     )

    # def forward(self, x):
    #     x = x + self.attn(self.norm1(x))
    #     x = x + self.mlp(self.norm2(x))
    #     return x

    def __init__(self, dim, num_heads, window_size=7, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim*mlp_ratio), dim)
        )
        self.window_size = window_size

    def forward(self, x, H, W, mask=None):
        B, L, C = x.shape
        H_orig, W_orig = H, W
        x = x.view(B, H, W, C)

        # Partition windows
        x_windows, H_pad, W_pad = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size*self.window_size, C)

        # Attention
        attn_windows = self.attn(self.norm1(x_windows), mask=mask)

        # Merge windows
        x = window_reverse(attn_windows, self.window_size, H_pad, W_pad)
        x = x[:, :H_orig, :W_orig, :]  # crop padding
        x = x.reshape(B, H_orig*W_orig, C)

        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


# --- Patch Merging ---
class PatchMerging(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.reduction = nn.Linear(input_dim * 4, output_dim, bias=False)
        self.norm = nn.LayerNorm(input_dim * 4)

    def forward(self, x, H, W):
        B, N, C = x.shape
        assert N == H * W
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()      # [B, C, H, W]
        x_unfold = F.unfold(x, kernel_size=2, stride=2)  # [B, 4*C, H/2*W/2]
        x_unfold = x_unfold.transpose(1, 2)              # [B, H/2*W/2, 4*C]
        x_unfold = self.norm(x_unfold)
        x_reduced = self.reduction(x_unfold)             # [B, H/2*W/2, output_dim]
        return x_reduced, H // 2, W // 2

# --- Swin Stage ---
# class SwinStage(nn.Module):
#     def __init__(self, dim, num_blocks, num_heads, window_size, patch_merge=False, output_dim=None):
#         super().__init__()
#         self.blocks = nn.ModuleList([SwinTransformerBlock(dim, num_heads, window_size) for _ in range(num_blocks)])
#         self.patch_merge = patch_merge
#         if patch_merge:
#             self.merge = PatchMerging(dim, output_dim)

#     def forward(self, x, H, W):
#         for block in self.blocks:
#             x = block(x)
#         if self.patch_merge:
#             x, H, W = self.merge(x, H, W)
#         return x, H, W

class SwinStage(nn.Module):
    def __init__(self, dim, num_blocks, num_heads, window_size, patch_merge = False, output_dim = None):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            # alternate shift for every other block
            self.blocks.append(SwinTransformerBlock(dim=dim, num_heads=num_heads,
                                                    window_size=window_size))
        self.patch_merge = patch_merge
        if patch_merge:
            assert output_dim is not None
            self.merge = PatchMerging(dim, output_dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        # prepare attention mask per block if needed (only for blocks with shift)
        device = x.device
        for blk in self.blocks:
            x = blk(x, H, W)

        if self.patch_merge:
            x, H, W = self.merge(x, H, W)
        return x, H, W

class SwinDETRBackbone(nn.Module):
    def __init__(self, embed_dim=128, num_heads=[4, 8, 16, 32], num_blocks=[2,2,6,2], window_size=7):
        super().__init__()
        self.embed_dim = embed_dim
        self.downsample_cnn = DownsampleCNN(in_channels=3, out_channels=embed_dim)

        # Stage 1: linear embeddings, no patch merging
        self.stage1 = SwinStage(embed_dim, num_blocks[0], num_heads[0], window_size, patch_merge=False)
        # Stage 2-4: with patch merging
        self.stage2 = SwinStage(embed_dim, num_blocks[1], num_heads[1], window_size, patch_merge=True, output_dim=embed_dim*2)
        self.stage3 = SwinStage(embed_dim*2, num_blocks[2], num_heads[2], window_size, patch_merge=True, output_dim=embed_dim*4)
        self.stage4 = SwinStage(embed_dim*4, num_blocks[3], num_heads[3], window_size, patch_merge=True, output_dim=embed_dim*8)

    def forward(self, x):
        B, _, H, W = x.shape
        # Step 1: CNN downsample
        x = self.downsample_cnn(x)  # [B, embed_dim, H/4, W/4]

        # Step 2: Flatten to tokens
        B, C, H, W = x.shape
        x_tokens = x.permute(0, 2, 3, 1).contiguous()  # [B, H*W, C]
        x_tokens = x_tokens.view(B, H * W, C)

        # Step 3: 4 Swin stages
        x_tokens, H, W = self.stage1(x_tokens, H, W)
        print("Stage1:", x_tokens.mean().item(), x_tokens.std().item(), "H=", H, "W=", W)

        x_tokens, H, W = self.stage2(x_tokens, H, W)
        print("Stage2:", x_tokens.mean().item(), x_tokens.std().item(), "H=", H, "W=", W)

        x_tokens, H, W = self.stage3(x_tokens, H, W)
        print("Stage3:", x_tokens.mean().item(), x_tokens.std().item(), "H=", H, "W=", W)

        x_tokens, H, W = self.stage4(x_tokens, H, W)
        print("Stage4:", x_tokens.mean().item(), x_tokens.std().item(), "H=", H, "W=", W)


        B, N, C_out = x_tokens.shape
        x_feat = x_tokens.view(B,H,W,C_out).permute(0,3,1,2).contiguous()
        return x_feat    # [B, C_out, H, W]

# --- Example usage ---
#x = torch.randn(6, 3, 224, 224)  # batch_size=6
#backbone = SwinDETRBackbone()
#features, H_out, W_out = backbone(x)
#print("Final features shape:", features.shape)  # [B, N_final, C_final]
#print("Final H x W:", H_out, W_out)

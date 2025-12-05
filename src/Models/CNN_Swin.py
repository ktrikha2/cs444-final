import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out

# --- Swin Transformer Block ---
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# --- Patch Merging ---
class PatchMerging(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.reduction = nn.Linear(input_dim * 4, output_dim, bias=False)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)      # [B, C, H, W]
        x_unfold = F.unfold(x, kernel_size=2, stride=2)  # [B, 4*C, H/2*W/2]
        x_unfold = x_unfold.transpose(1, 2)              # [B, H/2*W/2, 4*C]
        x_reduced = self.reduction(x_unfold)             # [B, H/2*W/2, output_dim]
        return x_reduced, H // 2, W // 2

# --- Swin Stage ---
class SwinStage(nn.Module):
    def __init__(self, dim, num_blocks, num_heads, window_size, patch_merge=False, output_dim=None):
        super().__init__()
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim, num_heads, window_size) for _ in range(num_blocks)])
        self.patch_merge = patch_merge
        if patch_merge:
            self.merge = PatchMerging(dim, output_dim)

    def forward(self, x, H, W):
        for block in self.blocks:
            x = block(x)
        if self.patch_merge:
            x, H, W = self.merge(x, H, W)
        return x, H, W

class SwinDETRBackbone(nn.Module):
    def __init__(self, embed_dim=128, num_heads=[4, 8, 16, 32], num_blocks=[2,2,6,2], window_size=7):
        super().__init__()
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
        x_tokens = x.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # Step 3: 4 Swin stages
        x, H, W = self.stage1(x_tokens, H, W)
        x, H, W = self.stage2(x, H, W)
        x, H, W = self.stage3(x, H, W)
        x, H, W = self.stage4(x, H, W)

        return x, H, W

# --- Example usage ---
x = torch.randn(6, 3, 224, 224)  # batch_size=6
backbone = SwinDETRBackbone()
features, H_out, W_out = backbone(x)
print("Final features shape:", features.shape)  # [B, N_final, C_final]
print("Final H x W:", H_out, W_out)

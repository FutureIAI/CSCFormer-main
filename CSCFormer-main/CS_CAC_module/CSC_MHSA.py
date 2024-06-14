import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Mlp(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.in_features = cfg
        self.hidden_features = cfg//2
        self.fc1 = nn.Linear(self.in_features, self.hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.hidden_features, self.in_features)
        self.drop = nn.Dropout(0.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, cfg, head_num):
        super().__init__()

        self.dim = cfg
        self.num_heads = head_num
        self.head_dim = self.dim // self.num_heads
        assert self.dim % self.num_heads == 0, f"dim {self.dim} should be divided by num_heads {self.num_heads}."
        self.scale = self.head_dim ** -0.5

        self.qkv1 = nn.Linear(self.dim, self.dim * 3)
        self.qkv2 = nn.Linear(self.dim, self.dim * 3)

        self.qkv3 = nn.Linear(self.dim, self.dim * 3)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=head_num*3,out_channels=head_num,stride=1,kernel_size=1,padding=0),
            nn.BatchNorm2d(head_num, 0.8),
            nn.LeakyReLU(0.2)
        )

        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x):
        B, N, C = x.shape
        feature1, feature2, feature3 = x.chunk(3, 1)

        qkv1 = self.qkv1(feature1).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]
        qkv2 = self.qkv2(feature2).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]

        qkv3 = self.qkv3(feature3).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q3, k3, v3 = qkv3[0], qkv3[1], qkv3[2]

        q = torch.cat([q1,q2,q3],dim=1)
        q = self.conv(q)

        attn2 = (q @ k2.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        v2 = (attn2 @ v2).transpose(1, 2).reshape(B, N // 3, C)

        attn1 = (q @ k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        v1 = (attn1 @ v1).transpose(1, 2).reshape(B, N // 3, C)

        attn3 = (q @ k3.transpose(-2, -1)) * self.scale
        attn3 = attn3.softmax(dim=-1)
        attn3 = self.attn_drop(attn3)
        v3 = (attn3 @ v3).transpose(1, 2).reshape(B, N // 3, C)

        x = torch.cat([v1,v2,v3], dim=1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, head_num, dim, dpr=0.5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = CrossAttention(dim, head_num)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.mlp = Mlp(dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

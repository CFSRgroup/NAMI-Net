import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info
from torch.nn import init
import math

# References & Acknowledgments
# Portions of the code in this file were adapted from the following source:
# https://github.com/JIAOJIAYUASD/dilateformer

class DilateAttention(nn.Module):
    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=dilation*(kernel_size-1)//2, stride=1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v):
        B, d, H, W = q.shape

        q = q.reshape(B, d // self.head_dim, self.head_dim, 1, H * W).permute(0, 1, 4, 3, 2).clone()

        k = self.unfold(k).clone()  # clone to avoid inplace operation
        k = k.reshape(B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W).permute(0, 1, 4, 2, 3).clone()

        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        v = self.unfold(v).clone()  # clone to avoid inplace operation
        v = v.reshape(B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W).permute(0, 1, 4, 3, 2).clone()

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d).clone()
        return x


class MultiDilatelocalAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0, proj_drop=0, kernel_size=3, dilation=[1,2,3,4]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.num_dilation = len(dilation)
        assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.dilate_attention = nn.ModuleList(
            [DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.num_dilation)])
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2).clone()

        qkv = self.qkv(x).reshape(B, 3, self.num_dilation, C // self.num_dilation, H, W).permute(2, 1, 0, 3, 4, 5).clone()
        x = x.reshape(B, self.num_dilation, C // self.num_dilation, H, W).permute(1, 0, 3, 4, 2).clone()

        for i in range(self.num_dilation):
            x[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2]).clone()  # clone to avoid inplace operation

        x = x.permute(1, 2, 3, 0, 4).reshape(B, H, W, C).clone()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DilateBlock(nn.Module):
    def __init__(self, attn_drop=0, proj_drop=0, dim=256):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.attn = MultiDilatelocalAttention(attn_drop=attn_drop, proj_drop=proj_drop, dim=dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = x + self.attn(self.norm(x))
        x = x.permute(0, 3, 1, 2)
        return x





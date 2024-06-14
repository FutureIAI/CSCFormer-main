import torch
import torch.nn as nn
from timm.models.layers import to_2tuple
from CSAF_module.ASFF_S import ASFF_TwoInput
from CS_CAC_module.Deepconv import DEPTHWISECONV
from thop import profile
from thop import clever_format

class CrossAttention(nn.Module):
    '''
    空间缩减交叉多头自注意力机制
    '''
    def __init__(self, cfg, head_num, sr_ratios):
        super().__init__()

        # 输入通道数
        self.dim = cfg
        # 输入头数
        self.num_heads = head_num
        self.head_dim = self.dim // self.num_heads
        # 空间缩减因子
        self.sr_ratio = sr_ratios

        assert self.dim % self.num_heads == 0, f"dim {self.dim} should be divided by num_heads {self.num_heads}."
        self.scale = self.head_dim ** -0.5

        #
        self.scale1_q = nn.Linear(self.dim, self.dim, bias=0.0)
        self.scale1_kv = nn.Linear(self.dim, self.dim * 2, bias=0.0)
        self.scale2_q = nn.Linear(self.dim, self.dim, bias=0.0)
        self.scale2_kv = nn.Linear(self.dim, self.dim * 2, bias=0.0)

        # 空间缩减模块
        if self.sr_ratio > 1:
            self.scale1_sr = DEPTHWISECONV(in_ch=self.dim,out_ch=self.dim,k=self.sr_ratio,s=self.sr_ratio)
            self.scale1_norm = nn.LayerNorm(self.dim)
            self.scale2_sr = DEPTHWISECONV(in_ch=self.dim,out_ch=self.dim,k=self.sr_ratio,s=self.sr_ratio)
            self.scale2_norm = nn.LayerNorm(self.dim)
            pass

        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(0.0)

        # 融合两个尺度的q的卷积
        self.conv = nn.Sequential(
            DEPTHWISECONV(in_ch=head_num * 2, out_ch=head_num, k=1, s=1,p=0),
            nn.BatchNorm2d(head_num, 0.8),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        scale1, scale2 = x.chunk(2, 1)
        scale1_q = self.scale1_q(scale1).reshape(B, N // 2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        scale2_q = self.scale2_q(scale2).reshape(B, N // 2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # 空间缩减模块
        if self.sr_ratio > 1:
            scale1_ = scale1.permute(0, 2, 1).reshape(B, C, H, W)
            scale1_ = self.scale1_sr(scale1_).reshape(B, C, -1).permute(0, 2, 1)
            scale1_ = self.scale1_norm(scale1_)
            scale1_kv = self.scale1_kv(scale1_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            scale2_ = scale2.permute(0, 2, 1).reshape(B, C, H, W)
            scale2_ = self.scale2_sr(scale2_).reshape(B, C, -1).permute(0, 2, 1)
            scale2_ = self.scale2_norm(scale2_)
            scale2_kv = self.scale2_kv(scale2_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                            4)
        else:
            scale1_kv = self.scale1_kv(scale1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            scale2_kv = self.scale2_kv(scale2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                    4)
        # 融合q卷积
        fusion_q = torch.cat([scale1_q, scale2_q], dim=1)
        fusion_q = self.conv(fusion_q)


        scale1_k, scale1_v = scale1_kv[0], scale1_kv[1]
        scale2_k, scale2_v = scale2_kv[0], scale2_kv[1]

        scale2_attn = (fusion_q @ scale2_k.transpose(-2, -1)) * self.scale
        scale2_attn = scale2_attn.softmax(dim=-1)
        scale2_attn = self.attn_drop(scale2_attn)
        scale2_v = (scale2_attn @ scale2_v).transpose(1, 2).reshape(B, N // 2, C)

        scale1_attn = (fusion_q @ scale1_k.transpose(-2, -1)) * self.scale
        scale1_attn = scale1_attn.softmax(dim=-1)
        scale1_attn = self.attn_drop(scale1_attn)
        scale1_v = (scale1_attn @ scale1_v).transpose(1, 2).reshape(B, N // 2, C)

        x = torch.cat([scale1_v, scale2_v], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
class CSAF(nn.Module):
    '''
    跨尺度自适应特征融合模块
    '''
    def __init__(self,cfg, head_num, sr_ratios,level):
        super(CSAF, self).__init__()

        # 输入通道数
        self.dim = cfg
        # 尺度类型编码
        self.fpn1_class = nn.Parameter(torch.zeros(size=(1, 1, self.dim//2)))
        self.fpn2_class = nn.Parameter(torch.zeros(size=(1, 1, self.dim//2)))
        self.fpn3_class = nn.Parameter(torch.zeros(size=(1, 1, self.dim//2)))

        # 位置编码
        self.pos_encode = nn.Conv2d(self.dim, self.dim, to_2tuple(3), to_2tuple(1), to_2tuple(1),
                                    groups=self.dim)

        self.ln_norm1 = nn.LayerNorm(self.dim//2, eps=1e-6)
        self.attn = CrossAttention(self.dim//2, head_num, sr_ratios)
        self.proj = nn.Conv2d(self.dim , self.dim, to_2tuple(1), bias=False)

        # 残差分支
        self.residual = nn.Sequential(
            nn.Conv2d(self.dim , self.dim, to_2tuple(1), bias=False),
            nn.Conv2d(self.dim, self.dim, to_2tuple(3), to_2tuple(1), to_2tuple(1), groups=self.dim,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim, self.dim, to_2tuple(1), bias=False),
            nn.BatchNorm2d(self.dim)
        )
        self.bn_norm = nn.BatchNorm2d(self.dim)

        # ASFF模块,输入两个尺度,输出两个尺度
        self.asff = ASFF_TwoInput(self.dim//2, level,False, False)

        # 减少通道数卷积
        self.reduce_channel_1 = nn.Conv2d(in_channels=self.dim,out_channels=self.dim//2,kernel_size=1,stride=1,padding=0)
        self.reduce_channel_2 = nn.Conv2d(in_channels=self.dim, out_channels=self.dim // 2, kernel_size=1, stride=1,
                                          padding=0)
        pass

    def forward(self,fpn1, fpn2):

        # 各自通道数将半
        fpn1 = self.reduce_channel_1(fpn1)
        fpn2 = self.reduce_channel_2(fpn2)

        # ASFF融合
        x1,x2 = self.asff(fpn1,fpn2)

        B, C, H, W = x1.shape
        x = torch.cat([x1, x2], dim=1)

        # 残差分支
        residual = x

        # 位置编码
        x = self.pos_encode(x) + x

        # 拆分得到两个不同尺度的特征
        fpn1 = x.chunk(2, 1)[0].reshape(B, C, -1).permute(0, 2, 1)
        fpn2 = x.chunk(2, 1)[1].reshape(B, C, -1).permute(0, 2, 1)

        # 为不同尺度的特征添加尺度类型编码
        x = torch.cat([fpn1 + self.fpn1_class, fpn2 + self.fpn2_class], dim=1)

        x = self.ln_norm1(self.attn(x, H, W) + x)
        x = x.permute(0, 2, 1).reshape(B, C * 2 , H, W)

        # 将残差分支的特征与主分支的特征相加
        out = self.bn_norm(self.proj(x) + self.residual(residual))
        return out
        pass
    pass



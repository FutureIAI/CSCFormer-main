import torch
import torch.nn as nn
import torch.nn.functional as F
from CS_CAC_module.Deepconv import DEPTHWISECONV

def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv',DEPTHWISECONV(in_ch=in_ch,out_ch=out_ch,k=ksize,s=stride,p=pad))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage

class SPPLayer(nn.Module):
    def __init__(self):
        super(SPPLayer, self).__init__()

    def forward(self, x):
        x_1 = x
        x_2 = F.max_pool2d(x, 5, stride=1, padding=2)
        x_3 = F.max_pool2d(x, 9, stride=1, padding=4)
        x_4 = F.max_pool2d(x, 13, stride=1, padding=6)
        out = torch.cat((x_1, x_2, x_3, x_4), dim=1)
        return out


class ASFFmobile(nn.Module):
    def __init__(self, level, rfb=False, vis=False):
        super(ASFFmobile, self).__init__()
        self.level = level
        self.dim = [512, 256, 128]
        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = add_conv(256, self.inter_dim, 3, 2, leaky=False)
            self.stride_level_2 = add_conv(128, self.inter_dim, 3, 2, leaky=False)
            self.expand = add_conv(self.inter_dim, 1024, 3, 1, leaky=False)
        elif level == 1:
            self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1, leaky=False)
            self.stride_level_2 = add_conv(128, self.inter_dim, 3, 2, leaky=False)
            self.expand = add_conv(self.inter_dim, 512, 3, 1, leaky=False)
        elif level == 2:
            self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1, leaky=False)
            self.compress_level_1 = add_conv(256, self.inter_dim, 1, 1, leaky=False)
            self.expand = add_conv(self.inter_dim, 256, 3, 1, leaky=False)

        compress_c = 8 if rfb else 16  # when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1, leaky=False)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1, leaky=False)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1, leaky=False)

        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        self.vis = vis

    def forward(self, x_level_0, x_level_1, x_level_2):
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)

            level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)

        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


class ASFF(nn.Module):
    def __init__(self, in_ch,level, rfb=False, vis=False):
        super(ASFF, self).__init__()
        self.level = level
        self.dim = [in_ch, in_ch, in_ch]
        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = add_conv(in_ch, self.inter_dim, 3, 2)
            self.stride_level_2 = add_conv(in_ch, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, in_ch, 3, 1)
        elif level == 1:
            self.compress_level_0 = add_conv(in_ch, self.inter_dim, 1, 1)
            self.stride_level_2 = add_conv(in_ch, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, in_ch, 3, 1)
        elif level == 2:
            self.compress_level_0 = add_conv(in_ch, self.inter_dim, 1, 1)
            self.expand = add_conv(self.inter_dim, in_ch, 3, 1)


        compress_c = 8 if rfb else 16  # when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_levels = nn.Sequential(
            nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        )

        self.vis = vis

    def forward(self, x_level_0, x_level_1, x_level_2):

        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_resized = F.interpolate(x_level_1, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2

        w0 = level_0_resized.shape[3]
        w1 = level_1_resized.shape[3]
        w2 = level_2_resized.shape[3]
        min_w = 10000
        if w0 < min_w:
            min_w = w0
            pass
        if w1 < min_w:
            min_w = w1
            pass
        if w2 < min_w:
            min_w = w2
            pass
        level_0_resized = level_0_resized[:, :, :, 0:min_w]
        level_1_resized = level_1_resized[:, :, :, 0:min_w]
        level_2_resized = level_2_resized[:, :, :, 0:min_w]

        h0 = level_0_resized.shape[2]
        h1 = level_1_resized.shape[2]
        h2 = level_2_resized.shape[2]
        min_h = 10000
        if h0 < min_h:
            min_h = h0
            pass
        if h1 < min_h:
            min_h = h1
            pass
        if h2 < min_h:
            min_h = h2
            pass
        level_0_resized = level_0_resized[:, :, 0:min_h, :]
        level_1_resized = level_1_resized[:, :, 0:min_h, :]
        level_2_resized = level_2_resized[:, :, 0:min_h, :]

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


class ASFF_CSAF(nn.Module):
    def __init__(self,in_ch,level, rfb=False, vis=False):
        super(ASFF_CSAF, self).__init__()
        self.level = level
        self.dim = [in_ch, in_ch, in_ch]
        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = add_conv(in_ch, self.inter_dim, 3, 2)
            self.stride_level_2 = add_conv(in_ch, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, in_ch*3, 3, 1)
        elif level == 1:
            self.compress_level_0 = add_conv(in_ch, self.inter_dim, 1, 1)
            self.stride_level_2 = add_conv(in_ch, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, in_ch*3, 3, 1)
        elif level == 2:
            self.compress_level_0 = add_conv(in_ch, self.inter_dim, 1, 1)
            self.expand = add_conv(self.inter_dim, in_ch*3, 3, 1)

        compress_c = 8 if rfb else 16  # when adding rfb, we use half number of channels to save memory
        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_levels = nn.Sequential(
            nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)

        )

        self.vis = vis
        pass
    def forward(self,x_level_0, x_level_1, x_level_2):
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_resized = F.interpolate(x_level_1, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2

        w0 = level_0_resized.shape[3]
        w1 = level_1_resized.shape[3]
        w2 = level_2_resized.shape[3]
        min_w = 10000
        if w0 < min_w:
            min_w = w0
            pass
        if w1 < min_w:
            min_w = w1
            pass
        if w2 < min_w:
            min_w = w2
            pass
        level_0_resized = level_0_resized[:, :, :, 0:min_w]
        level_1_resized = level_1_resized[:, :, :, 0:min_w]
        level_2_resized = level_2_resized[:, :, :, 0:min_w]

        h0 = level_0_resized.shape[2]
        h1 = level_1_resized.shape[2]
        h2 = level_2_resized.shape[2]
        min_h = 10000
        if h0 < min_h:
            min_h = h0
            pass
        if h1 < min_h:
            min_h = h1
            pass
        if h2 < min_h:
            min_h = h2
            pass
        level_0_resized = level_0_resized[:, :, 0:min_h, :]
        level_1_resized = level_1_resized[:, :, 0:min_h, :]
        level_2_resized = level_2_resized[:, :, 0:min_h, :]

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
        return (level_0_resized * levels_weight[:, 0:1, :, :], level_1_resized * levels_weight[:, 1:2, :, :],
                level_2_resized * levels_weight[:, 2:, :, :])
        pass

    pass


class ASFF_TwoInput(nn.Module):
    def __init__(self,in_ch,level, rfb=False, vis=False):
        super(ASFF_TwoInput, self).__init__()
        self.level = level
        self.dim = [in_ch, in_ch, in_ch]
        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = add_conv(in_ch, self.inter_dim, 3, 2)
            self.stride_level_2 = add_conv(in_ch, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, in_ch * 3, 3, 1)
        elif level == 1:
            self.compress_level_0 = add_conv(in_ch, self.inter_dim, 1, 1)
            self.stride_level_2 = add_conv(in_ch, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, in_ch * 3, 3, 1)


        compress_c = 8 if rfb else 16  # when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Sequential(
            nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)
        )

        self.vis = vis
        pass
    def forward(self,x_level_0, x_level_1):
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1

        w0 = level_0_resized.shape[3]
        w1 = level_1_resized.shape[3]

        min_w = 10000
        if w0 < min_w:
            min_w = w0
            pass
        if w1 < min_w:
            min_w = w1
            pass

        level_0_resized = level_0_resized[:, :, :, 0:min_w]
        level_1_resized = level_1_resized[:, :, :, 0:min_w]


        h0 = level_0_resized.shape[2]
        h1 = level_1_resized.shape[2]

        min_h = 10000
        if h0 < min_h:
            min_h = h0
            pass
        if h1 < min_h:
            min_h = h1
            pass
        level_0_resized = level_0_resized[:, :, 0:min_h, :]
        level_1_resized = level_1_resized[:, :, 0:min_h, :]

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        return (level_0_resized * levels_weight[:, 0:1, :, :], level_1_resized * levels_weight[:, 1:2, :, :])
        pass

    pass


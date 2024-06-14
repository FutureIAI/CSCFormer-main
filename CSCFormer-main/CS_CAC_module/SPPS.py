import torch
import torch.nn as nn
from einops import rearrange

class SPP(nn.Module):
    def __init__(self):
        super(SPP, self).__init__()
        self.target_size = [(1, 1), (3, 4), (6, 8), (12, 16)]
        for i in range(len(self.target_size)):
            setattr(self, 'avg_pool{}'.format(i), nn.AdaptiveAvgPool2d(self.target_size[i]))
            setattr(self, 'max_pool{}'.format(i), nn.AdaptiveMaxPool2d(self.target_size[i]))

    def forward(self, x):
        avg_scales = []
        max_scales = []
        for i in range(len(self.target_size)):
            avg_scale = getattr(self, 'avg_pool{}'.format(i))(x)
            max_scale = getattr(self, 'max_pool{}'.format(i))(x)
            avg_scale = rearrange(avg_scale, 'b c h w -> b c (h w)')
            max_scale = rearrange(max_scale, 'b c h w -> b c (h w)')
            avg_scales.append(avg_scale)
            max_scales.append(max_scale)
        x = torch.cat(avg_scales + max_scales, dim=2)
        return x
import torch
import torch.nn.functional as F
import torch.nn as nn
from CS_CAC_module.Deepconv import DEPTHWISECONV,TransDEPTHWISECONV


class FPN(nn.Module):
    def __init__(self, in_channel_list, out_channel):
        super(FPN, self).__init__()

        self.inner_layer = nn.ModuleList()
        self.out_layer = nn.ModuleList()

        for in_channel in in_channel_list:
            self.inner_layer.append(DEPTHWISECONV(in_channel,out_channel,1,1,0,norm=False))
            self.out_layer.append(DEPTHWISECONV(out_channel,out_channel,3,1,1,norm=False))

    def forward(self, x):
        head_output = []
        corent_inner = self.inner_layer[-1](x[-1])
        head_output.append(self.out_layer[-1](corent_inner))
        for i in range(len(x) - 2, -1, -1):
            pre_inner = corent_inner
            corent_inner = self.inner_layer[i](x[i])
            size = corent_inner.shape[2:]
            pre_top_down = F.interpolate(pre_inner, size=size)
            add_pre2corent = pre_top_down + corent_inner
            head_output.append(self.out_layer[i](add_pre2corent))
        return list(reversed(head_output))


from CS_CAC_module.SPPS import SPP
from CS_CAC_module.CSC_MHSA import Block
from CS_CAC_module.Deepconv import DEPTHWISECONV,TransDEPTHWISECONV
import torch
import torch.nn as nn
from CS_CAC_module.FPN import FPN
from CSAF_module.ASFF_S import ASFF
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Cross_Branch(nn.Module):
    def __init__(self, head_num, dims, block_num,in_channel):
        super(Cross_Branch, self).__init__()
        #
        self.spp = SPP()
        self.proj = nn.Linear(in_features=506, out_features=dims)
        self.proj2 = nn.Sequential(
            nn.Linear(dims, dims),
            nn.ReLU(inplace=True),
            nn.Linear(dims, 1),
            nn.Sigmoid()
        )

        self.blocks_1 = block_num

        self.blocks_2 = block_num

        self.blocks_3 = block_num

        # 注意力机制模块
        self.mhsas = nn.ModuleList([Block(head_num, dims)
                                      for j in range(block_num)])

        #
        self.del_channel_conv_1 = nn.Linear(in_features=in_channel * 2, out_features=in_channel)
        self.del_channel_conv_2 = nn.Linear(in_features=in_channel * 4, out_features=in_channel)
        #
        self.del_channel_conv_3 = nn.Linear(in_features=in_channel * 3, out_features=in_channel * 2)
        self.del_channel_conv_4 = nn.Linear(in_features=in_channel * 3, out_features=in_channel)
        self.del_channel_conv_5 = nn.Linear(in_features=in_channel * 3, out_features=in_channel * 4)

        # 尺度类型编码
        self.large_scale_class = nn.Parameter(torch.zeros(size=(1,1,dims)))
        self.medium_scale_class = nn.Parameter(torch.zeros(size=(1, 1, dims)))
        self.small_scale_class = nn.Parameter(torch.zeros(size=(1, 1, dims)))

        pass
    def forward(self,features):
        feature1 = features[0]
        feature2 = features[1]
        feature3 = features[2]

        # 第一分支
        feature1 = self.spp(feature1)  # SPP：例如(1,96,192,192)--->(1，96，506)
        feature1 = self.proj(feature1)  # 投影: 把506投影成dims=[512, 320, 128, 64],例如(1，96，506)--->(1，96，512)
        feature1 = feature1 + self.large_scale_class

        # 第二分支
        feature2 = self.spp(feature2)  # SPP：例如(1,96,192,192)--->(1，96，506)
        feature2 = self.proj(feature2)  # 投影: 把506投影成dims=[512, 320, 128, 64],例如(1，96，506)--->(1，96，512)
        feature2 = feature2 + self.medium_scale_class

        # 第三分支
        feature3 = self.spp(feature3)  # SPP：例如(1,96,192,192)--->(1，96，506)
        feature3 = self.proj(feature3)  # 投影: 把506投影成dims=[512, 320, 128, 64],例如(1，96，506)--->(1，96，512)
        feature3 = feature3 + self.small_scale_class

        feature_cat = torch.cat([feature1, feature2, feature3], dim=1)

        for block in self.mhsas:  # Transformer block,例(1，96，512)--->(1，96，512)
            feature_cat = block(feature_cat)

        feature1 = self.del_channel_conv_4(feature_cat.permute(0,2,1)).permute(0,2,1)
        feature2 = self.del_channel_conv_3(feature_cat.permute(0,2,1)).permute(0,2,1)
        feature3 = self.del_channel_conv_5(feature_cat.permute(0,2,1)).permute(0,2,1)

        # 第一分支
        feature1 = self.proj2(feature1).permute(0, 2, 1)  # 投影: 例如把(1,96,512)--->(1,1,96) 变为一维向量

        # 第二分支
        feature2 = self.proj2(feature2).permute(0, 2, 1)  # 投影: 例如把(1,96,512)--->(1,1,192) 变为一维向量

        # 第三分支
        feature3 = self.proj2(feature3).permute(0, 2, 1)  # 投影: 例如把(1,96,512)--->(1,1,384) 变为一维向量

        return (feature1, feature2, feature3)
        pass

class CS_CAC(nn.Module):
    def __init__(self, in_channel, stage):
        super(CS_CAC, self).__init__()

        self.in_channel = in_channel

        # 下采样
        self.downsample_1 = DEPTHWISECONV(in_channel, in_channel * 2, 4, 2, 1, norm=True)
        self.downsample_2 = DEPTHWISECONV(in_channel * 2, in_channel * 4, 4, 2, 1, norm=True)

        self.fpn_scale = [in_channel,in_channel * 2,in_channel * 4]
        self.ouput_fpn = in_channel
        self.fpn = FPN(self.fpn_scale, self.ouput_fpn)  # FPN

        self.scale_dim = [512, 320, 128, 64]  # 每个stage的输入到CS_CAC模块的尺度大小
        self.heads = [8, 5, 2, 1]  # 每个stage的输入到CS_CAC模块的头数

        self.bloks = [2, 2, 2, 2]  # CS_CAC模块在每个stage的输入的block数量           #MS_T 尺寸型号
        # self.bloks = [3, 4, 6, 3]  # CS_CAC模块在每个stage的输入的block数量             #MS_S 中尺寸型号
        # self.bloks = [3, 4, 18, 3]  # CS_CAC模块在每个stage的输入的block数量             #MS_B 大尺寸型号

        self.cross_CSCAC = Cross_Branch(self.heads[stage], self.scale_dim[stage], self.bloks[stage],in_channel)

        self.asff = ASFF(in_channel,2,False,False)

        # 下采样
        self.dowmsample_3 = DEPTHWISECONV(in_channel, in_channel * 2, 4, 2, 1, norm=True)

        self.channel_conv_1 = nn.Conv2d(in_channels=in_channel*2,out_channels=in_channel,kernel_size=1,stride=1)
        self.channel_conv_2 = nn.Conv2d(in_channels=in_channel*4,out_channels=in_channel,kernel_size=1,stride=1)

        # 适应输入大小
        self.transConv_H = TransDEPTHWISECONV(in_ch=in_channel,out_ch=in_channel,k=3,s=1,p=(1,0))
        self.transConv_W = TransDEPTHWISECONV(in_ch=in_channel, out_ch=in_channel, k=3, s=1, p=(0, 1))
        self.isNotEqual_W = False
        self.isNotEqual_H = False
        pass

    def forward(self, x):

        feature_1 = x  # 原始stage的输出
        feature_2 = self.downsample_1(x)  # 第一次下采样, 例(1,96,192,192)--->(1,96*2,96,96)
        feature_3 = self.downsample_2(feature_2)  # 第二次下采样, 例(1,96*2,96,96)--->(1,96*4,48,48)

        fpn_input = [feature_1,feature_2,feature_3]

        features = self.fpn(fpn_input)

        output_CSCAC= self.cross_CSCAC(features)
        output_CSCAC_1 = output_CSCAC[0]
        output_CSCAC_2 = output_CSCAC[1]
        output_CSCAC_3 = output_CSCAC[2]

        fpn_0_C = feature_1.shape[1]
        fpn_0_W = feature_1.shape[2]
        fpn_0_H = feature_1.shape[3]
        fpn_1_C = feature_2.shape[1]
        fpn_1_W = feature_2.shape[2]
        fpn_1_H = feature_2.shape[3]
        fpn_2_C = feature_3.shape[1]
        fpn_2_W = feature_3.shape[2]
        fpn_2_H = feature_3.shape[3]

        feature_1 = feature_1.reshape(feature_1.shape[0], feature_1.shape[2] * feature_1.shape[3], feature_1.shape[1])
        feature_2 = feature_2.reshape(feature_2.shape[0], feature_2.shape[2] * feature_2.shape[3], feature_2.shape[1])
        feature_3 = feature_3.reshape(feature_3.shape[0], feature_3.shape[2] * feature_3.shape[3], feature_3.shape[1])

        output_CSCAC_1 = output_CSCAC_1 * feature_1
        output_CSCAC_2 = output_CSCAC_2 * feature_2

        output_CSCAC_3 = output_CSCAC_3 * feature_3

        output_CSCAC_1 = output_CSCAC_1.permute(0, 2, 1).reshape(output_CSCAC_1.shape[0], fpn_0_C, fpn_0_W, fpn_0_H)
        output_CSCAC_2 = output_CSCAC_2.permute(0, 2, 1).reshape(output_CSCAC_2.shape[0], fpn_1_C, fpn_1_W, fpn_1_H)
        output_CSCAC_3 = output_CSCAC_3.permute(0, 2, 1).reshape(output_CSCAC_3.shape[0], fpn_2_C, fpn_2_W, fpn_2_H)

        output_CSCAC_2 = self.channel_conv_1(output_CSCAC_2)
        output_CSCAC_3 = self.channel_conv_2(output_CSCAC_3)

        # 适应输入大小
        original_W = output_CSCAC_1.shape[2]
        original_H = output_CSCAC_1.shape[3]
        result = self.asff(output_CSCAC_3, output_CSCAC_2, output_CSCAC_1)
        asff_W = result.shape[2]
        asff_H = result.shape[3]
        if original_W != asff_W:
            result = self.transConv_W(result)
            pass
        if original_H != asff_H:
            result = self.transConv_H(result)
            pass

        W = result.shape[2]
        H = result.shape[3]
        C = result.shape[1]

        result = self.dowmsample_3(result)

        return result.permute(0, 2, 3, 1).view(result.shape[0], (W//2)*(H//2) , C * 2)
        pass



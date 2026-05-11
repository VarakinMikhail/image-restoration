import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from project_folder.layers import DWTLayer, IWTLayer

class ChannelAttention(nn.Module):
    def __init__(self, num_feat, reduction=16, bias=True):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mid_channels = max(1, num_feat // reduction)
        self.conv_du = nn.Sequential(
            nn.Conv2d(num_feat, mid_channels, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_feat, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RCAB(nn.Module):
    def __init__(self, num_feat, ca_reduction=16, bias=True):
        super(RCAB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=bias),
            ChannelAttention(num_feat, reduction=ca_reduction, bias=bias)
        )

    def forward(self, x):
        return x + self.body(x)

class RCAGroup(nn.Module):
    def __init__(self, num_feat, num_blocks, ca_reduction=16, bias=True):
        super(RCAGroup, self).__init__()
        modules = [RCAB(num_feat, ca_reduction, bias) for _ in range(num_blocks)]
        self.body = nn.Sequential(*modules)
        self.last_conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=bias)

    def forward(self, x):
        res = self.body(x)
        res = self.last_conv(res)
        return x + res

class ResGFMBlock(nn.Module):
    def __init__(self, num_feat, inter_ratio=2, bias=True):
        super(ResGFMBlock, self).__init__()
        
        inter_channels = max(1, int(num_feat / inter_ratio))
        
        self.cond_net = nn.Sequential(
            nn.Conv2d(num_feat, inter_channels, 3, 1, 1, bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(inter_channels, num_feat * 2, 1, bias=bias) 
        )
        
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=bias)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=bias)

    def forward(self, x):
        cond = self.cond_net(x)
        scale, shift = torch.chunk(cond, 2, dim=1)
        
        out = self.conv1(x)
        out = out * (scale + 1) + shift
        out = self.act(out)
        out = self.conv2(out)
        
        return x + out

class WaveletFusionNet(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 out_channels=3,
                 base_width=12,       
                 depth=2,             
                 enc_blk_nums=2,      
                 dec_blk_nums=2,      
                 use_gfm_in_enc=True,
                 gfm_inter_ratio=2.0,
                 ca_reduction=8, 
                 bias=True
                 ):
        super(WaveletFusionNet, self).__init__()
        
        self.depth = depth
        
        self.head_conv = nn.Conv2d(in_channels, base_width, 3, 1, 1, bias=bias)
        
        self.down_layers = nn.ModuleList()
        current_width = base_width
        
        for i in range(depth):
            dwt = DWTLayer()
            
            target_width = current_width * 2
            reduce_conv = nn.Conv2d(current_width * 4, target_width, 1, bias=bias)
            
            if use_gfm_in_enc:
                blocks = nn.Sequential(*[
                    ResGFMBlock(target_width, inter_ratio=gfm_inter_ratio, bias=bias) 
                    for _ in range(enc_blk_nums)
                ])
            else:
                blocks = RCAGroup(target_width, enc_blk_nums, ca_reduction=ca_reduction, bias=bias)
                
            self.down_layers.append(nn.Sequential(
                dwt,
                reduce_conv,
                blocks
            ))
            current_width = target_width
            
        self.bottleneck = RCAGroup(current_width, dec_blk_nums, ca_reduction=ca_reduction, bias=bias)
        
        self.up_layers = nn.ModuleList()
        self.fusion_layers = nn.ModuleList()
        
        for i in range(depth):
            target_width = current_width // 2
            
            expand_conv = nn.Conv2d(current_width, target_width * 4, 1, bias=bias)
            iwt = IWTLayer()
            
            self.up_layers.append(nn.Sequential(
                expand_conv,
                iwt
            ))
            
            self.fusion_layers.append(nn.Sequential(
                nn.Conv2d(target_width * 2, target_width, 1, bias=bias),
                RCAGroup(target_width, dec_blk_nums, ca_reduction=ca_reduction, bias=bias)
            ))
            
            current_width = target_width

        self.tail_conv = nn.Conv2d(base_width, out_channels, 3, 1, 1, bias=bias)

    def forward(self, x):
        identity = x
        feats = self.head_conv(x)
        
        skips = []
        skips.append(feats)
        
        for layer in self.down_layers:
            feats = layer(feats)
            skips.append(feats)
            
        res = self.bottleneck(feats)
        
        skips.pop() 
        
        for i in range(self.depth):
            res = self.up_layers[i](res)
            skip_feat = skips.pop()
            res_concat = torch.cat([res, skip_feat], dim=1)
            res = self.fusion_layers[i](res_concat)
            
        out = self.tail_conv(res)
        return identity + out
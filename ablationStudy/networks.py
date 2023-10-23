from pickle import FALSE, TRUE
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class DTPnet(nn.Module):
    def __init__(self, recurrent_iter=3, use_GPU=True, factor=10e-3, color=True, burst_length=1, blind_est=True, kernel_size=[5], sep_conv=False,
                 channel_att=False, spatial_att=False, upMode='bilinear', core_bias=False):
        super(DTPnet, self).__init__()
        # =========================================== Layer 1 ==================================
        dim = 64
        inp_channels=3
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim) # 先进行一次扩张
        # ============================================ Layer 2 ===================================
        # == Pre-Refeine
        out_channel_color = 3
        num_blocks = [4,6,6,8]
        num_refinement_blocks = 4
        heads = [1,2,4,8]

        ffn_expansion_factor=2.66
        bias=False
        LayerNorm_type='WithBias'
        # == Pre-Network
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.reduce_chan_level1 = nn.Conv2d(int(dim*2**1), int(dim), kernel_size=1, bias=bias)

        # == Other Parameters
        self.upMode = upMode
        self.burst_length = burst_length
        self.core_bias = core_bias
        self.color_channel = 3 if color else 1
        in_channel = (3 if color else 1) * (burst_length if blind_est else burst_length+1)
        out_channel = (3 if color else 1) * (2 * sum(kernel_size) if sep_conv else np.sum(np.array(kernel_size) ** 2)) * burst_length
        if core_bias:
            out_channel += (3 if color else 1) * burst_length

        # == Each Conv Layer
        # 2~5层都是均值池化+3层卷积 
        self.conv1_2 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])]) # ([18, 3, 96, 96]) -> ([18, 64, 96, 96])  in_channel, 64, channel_att=False, spatial_att=False
        self.conv2_3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])]) # 64, 128, channel_att=False, spatial_att=False
        self.conv3_4 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])]) # 128, 256, channel_att=False, spatial_att=False
        self.conv4_L = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])]) # 256, 512, channel_att=False, spatial_att=False
        # 基于空间注意力的上采样
        self.conv4_3_HSI = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])]) # 512+512, 512, channel_att=channel_att, spatial_att=TRUE
        self.conv3_2_HSI = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])]) # 256+512, 256, channel_att=channel_att, spatial_att=TRUE
        self.conv2_1_HSI = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])]) # 256+128, out_channel, channel_att=channel_att, spatial_att=TRUE

        # self.outc = nn.Conv2d(out_channel, out_channel, 1, 1, 0) # 0改为1
        # self.conv_final = nn.Conv2d(in_channels=75, out_channels=3, kernel_size=3, stride=1, padding=1) # 抛弃 KernelConv 将 75 channel 变为 12 channel
        self.outc = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.conv_final = nn.Conv2d(int(dim), out_channel_color, kernel_size=3, stride=1, padding=1, bias=bias)

        #============================================ Layer 3 ===================================
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), # 这个卷积完成后，张量会成比例缩小(第1次大概/10)
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), # 这个张良完成后，张量会恢复原大小
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), # 这个张量完成后，张量不但缩小/10，而且会产生负数(Tanh)
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1), # 这个张量完成后，张量保持和g输出的相似
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1), # 细节最后一层没有激活函数！2022.08.11 Jack
            )


    def forward(self, input):
        # ============================================ Layer 1 ===================================
        # Method 1 全归一化流
        input_level1 = self.patch_embed(input)  # torch.Size([18, 3, 96, 96]) ->  torch.Size([18, 64, 96, 96])  先进行一次扩张

        # ============================================ Layer 2.1 (mixTrans)===================================
        # 下采样
        conv1_2 = self.conv1_2(input_level1) # INV  torch.Size([18, 64, 96, 96]) ->  torch.Size([18, 64, 96, 96]) 
        input_level2 = self.down1_2(conv1_2) # torch.Size([18, 64, 96, 96]) -> torch.Size([18, 128, 48, 48])   # 下采样 block 1

        conv2_3 = self.conv2_3(input_level2) # INV  torch.Size([18, 128, 48, 48]) ->  torch.Size([18, 128, 48, 48]) 
        input_level3 = self.down2_3(conv2_3) # torch.Size([18, 128, 48, 48]) -> torch.Size([18, 256, 24, 24]) 

        conv3_4 = self.conv3_4(input_level3) # INV  torch.Size([18, 256, 24, 24]) ->  torch.Size([18, 256, 24, 24]) 
        input_level4 = self.down3_4(conv3_4) # torch.Size([18, 256, 24, 24]) -> torch.Size([18, 512, 12, 12])
        
        # 中间层
        conv4_L = self.conv4_L(input_level4) # INV  torch.Size([18, 512, 12, 12]) ->  torch.Size([18, 512, 12, 12]) 
        
        # 上采样
        inp_dec_level3 = self.up4_3(conv4_L) # torch.Size([6, 512, 12, 12]) -> torch.Size([6, 256, 24, 24]) size*2
        inp_dec_level3 = torch.cat([inp_dec_level3, conv3_4], 1) # torch.Size([6, 256, 24, 24]) -> torch.Size([6, 512, 24, 24]) concat: channel*2
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3) # torch.Size([6, 512, 24, 24]) -> torch.Size([6, 256, 24, 24]) channel/2
        conv4_3 = self.conv4_3_HSI(inp_dec_level3) # torch.Size([6, 256, 24, 24]) -> torch.Size([6, 256, 24, 24])  INV                 
        # 多尺度耦合1              
        conv4_3 = conv4_3 + self.up4_3(input_level4) # input_level4.torch.Size([6, 512, 12, 12])

        inp_dec_level2 = self.up3_2(conv4_3) # torch.Size([6, 256, 24, 24]) -> torch.Size([6, 128, 48, 48])
        inp_dec_level2 = torch.cat([inp_dec_level2, conv2_3], 1) # torch.Size([6, 128, 48, 48]) -> torch.Size([6, 256, 48, 48]) concat: channel*2
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2) # torch.Size([6, 256, 48, 48]) -> torch.Size([6, 128, 48, 48]) channel/2
        conv3_2 = self.conv3_2_HSI(inp_dec_level2) # ([6, 128, 48, 48]) -> ([6, 128, 48, 48])  INV              
        # 多尺度耦合2              
        conv3_2 = conv3_2 + self.up3_2(input_level3) # input_level3.torch.Size([6, 256, 24, 24])

        inp_dec_level1 = self.up2_1(conv3_2) # torch.Size([6, 128, 48, 48]) -> torch.Size([6, 64, 96, 96])
        inp_dec_level1 = torch.cat([inp_dec_level1, conv1_2], 1) # torch.Size([6, 64, 96, 96]) -> torch.Size([6, 128, 96, 96]) concat: channel*2
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1) # torch.Size([6, 128, 96, 96]) -> torch.Size([6, 64, 96, 96]) channel/2  # 源码的 Restormer 此处没有 reduce_chan_level1
        conv2_1 = self.conv2_1_HSI(inp_dec_level1) # torch.Size([6, 64, 96, 96]) -> torch.Size([6, 64, 96, 96])  INV   
        # 多尺度耦合3              
        conv2_1 = conv2_1 + self.up2_1(input_level2) # input_level2.torch.Size([6, 128, 48, 48])

        # 后处理
        core1 = self.outc(conv2_1) # conv2_1.torch.Size([6, 64, 96, 96]) ->  core1.torch.Size([6, 64, 96, 96])
        pred1 = self.conv_final(core1) # pred1.torch.Size([6, 3, 96, 96])

        #============================================ Layer 3 ===================================
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = pred1
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        x_list = []
        for count in range(self.iteration):
            x = torch.cat((input, x), 1) # 一次扩张
            x = self.conv0(x)

            x = torch.cat((x, h), 1) # 二次扩张
            i = self.conv_i(x) 
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g # 三次扩张：lstm的长记忆更新公式  
            h = o * torch.tanh(c) # lstm的短时记忆更新公式

            x = h #　以下逐级扩张
            resx = x
            x = F.relu(self.res_conv1(x) + resx) # 此次更新后x又缩小一个量级
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)  # BUG memory try setting max_split_size_mb to avoid fragmentation. 游戏本报错了！显存溢出！3.28 需要将batch改小点！
            x = self.conv(x) # 没有激活函数的这一层，数据瞬间变回正常量级！

            x = x + input
            x_list.append(x) # 只需要最终输出的x即可，x_list仅做一个记录

        # return x, count
        return x, pred1

#============================================================================== Transformer Block ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange



##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c') # x.shape = torch.Size([2, 3, 480, 320]) 将 长宽维度 合并 torch.Size([2, 153600, 3])

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module): # dim=48  LayerNorm_type='WithBias'
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True) # 经过 rearrange之后 x.torch.Size([2, 153600, 3])  mu.torch.Size([2, 153600, 1])
        sigma = x.var(-1, keepdim=True, unbiased=False) 
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias # torch.Size([2, 153600, 48]) * torch.Size([48]) + torch.Size([48])


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim) # dim=48  LayerNorm_type='WithBias'
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x): # x.torch.Size([2, 3, 480, 320])
        h, w = x.shape[-2:] # h,w torch.Size([480, 320])
        return to_4d(self.body(to_3d(x)), h, w) # 先压缩长宽 -> 再进行归一化缩放 -> 最后转回原来的shape



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x) # x.torch.Size([2, 48, 480, 320]) -> x.torch.Size([2, 254, 480, 320])
        x1, x2 = self.dwconv(x).chunk(2, dim=1) # x1.torch.Size([2, 127, 480, 320])    x2.torch.Size([2, 127, 480, 320])  chunk函数会将输入张量(input)沿着指定维度(dim)均匀的分割成特定数量的张量块(chunks)
        x = F.gelu(x1) * x2  
        x = self.project_out(x) 
        return x # x.torch.Size([2, 48, 480, 320])



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)  # .chunk()方法能够按照某维度,对张量进行均匀切分,并且返回结果是原张量的视图
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out # out.torch.Size([2, 48, 480, 320])


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type) # dim=48  LayerNorm_type='WithBias'
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x)) # ([2, 48, 480, 320]) -> torch.Size([2, 48, 480, 320])
        x = x + self.ffn(self.norm2(x)) # 

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv=====================Jack Propose: pyramid decomposition
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=64, bias=False): # torch.Size([6, 3, 96, 96])
        super(OverlapPatchEmbed, self).__init__()
        self.transform1 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(),
        )
        self.up = nn.Upsample(size=None, scale_factor=2, mode='bilinear', align_corners=False)
        self.transform2 = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x): # x.torch.Size([2, 3, 480, 320])
        x_real = self.transform2(x)
        x_conv = self.transform1(x)
        x_down_1 = F.interpolate(x_conv, scale_factor=0.5, mode='bilinear')
        x_down_2 = F.interpolate(x_down_1, scale_factor=0.5, mode='bilinear')
        x_up = self.up(x_down_2)
        x_final = x_down_1 - x_up # 可视化此处 === 加入该模块的原理：可以执行多尺度特征分解和组合，提供了一个初始的多尺度特征分解效果  把纹理细节提取出来了
        x_final = self.transform2(self.up(x_final))
        x_out = x_real - x_final # 减去纹理细节，获得雨条纹
        return x_out # x.torch.Size([2, 48, 480, 320])


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)



















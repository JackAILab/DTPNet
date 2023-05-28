from pickle import FALSE, TRUE
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np 
import numbers
from pdb import set_trace as stx
from einops import rearrange

class CTPnet(nn.Module):
    def __init__(self, recurrent_iter=3, use_GPU=True, factor=10e-3, color=True, burst_length=1, blind_est=True, kernel_size=[5], sep_conv=False,
                 channel_att=False, spatial_att=False, upMode='bilinear', core_bias=False):
        super(CTPnet, self).__init__()
        dim = 64
        inp_channels=3
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        out_channel_color = 3
        num_blocks = [4,6,6,8]
        num_refinement_blocks = 4
        heads = [1,2,4,8]

        ffn_expansion_factor=2.66
        bias=False
        LayerNorm_type='WithBias'


    def forward(self, input):

        input_level1 = self.patch_embed(input)  
        conv1_2 = self.conv1_2(input_level1) 
        input_level2 = self.down1_2(conv1_2) 

        conv2_3 = self.conv2_3(input_level2) 
        input_level3 = self.down2_3(conv2_3) 

        conv3_4 = self.conv3_4(input_level3) 
        input_level4 = self.down3_4(conv3_4) 
        
        conv4_L = self.conv4_L(input_level4) 
        
        inp_dec_level3 = self.up4_3(conv4_L) 
        inp_dec_level3 = torch.cat([inp_dec_level3, conv3_4], 1) 
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3) 
        conv4_3 = self.conv4_3_HSI(inp_dec_level3) 
        conv4_3 = conv4_3 + self.up4_3(input_level4) 

        inp_dec_level2 = self.up3_2(conv4_3) 
        inp_dec_level2 = torch.cat([inp_dec_level2, conv2_3], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        conv3_2 = self.conv3_2_HSI(inp_dec_level2)
        conv3_2 = conv3_2 + self.up3_2(input_level3) 

        inp_dec_level1 = self.up2_1(conv3_2) 
        inp_dec_level1 = torch.cat([inp_dec_level1, conv1_2], 1) 
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1) 
        conv2_1 = self.conv2_1_HSI(inp_dec_level1) 
        conv2_1 = conv2_1 + self.up2_1(input_level2) 

        core1 = self.outc(conv2_1) 
        pred1 = self.conv_final(core1) 

        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = pred1
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        x_list = []
        for count in range(self.iteration):
            x = torch.cat((input, x), 1) 
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x) 
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g 
            h = o * torch.tanh(c) 

            x = h 
            resx = x
            x = F.relu(self.res_conv1(x) + resx) 
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx) 
            x = self.conv(x)

            x = x + input
            x_list.append(x)

        # return x, count
        return x, pred1






def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c') 

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

class WithBias_LayerNorm(nn.Module): 
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
        mu = x.mean(-1, keepdim=True) 
        sigma = x.var(-1, keepdim=True, unbiased=False) 
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias 


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim) 
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:] 
        return to_4d(self.body(to_3d(x)), h, w) 


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x) 
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2  
        x = self.project_out(x) 
        return x 


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
        q,k,v = qkv.chunk(3, dim=1)  
        
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
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type) 
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x)) 
        x = x + self.ffn(self.norm2(x)) 

        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=64, bias=False): 
        super(OverlapPatchEmbed, self).__init__()
        self.transform1 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(),
        )
        self.up = nn.Upsample(size=None, scale_factor=2, mode='bilinear', align_corners=False)
        self.transform2 = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x): 
        x_real = self.transform2(x)
        x_conv = self.transform1(x)
        x_down_1 = F.interpolate(x_conv, scale_factor=0.5, mode='bilinear')
        x_down_2 = F.interpolate(x_down_1, scale_factor=0.5, mode='bilinear')
        x_up = self.up(x_down_2)
        x_final = x_down_1 - x_up 
        x_final = self.transform2(self.up(x_final))
        x_out = x_real - x_final 
        return x_out 


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


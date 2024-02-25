import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from einops import rearrange
import torch.nn.init as init

from models.id_embedding.iresnet import iresnet100, iresnet50

def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


# ############### Copied from StyleGAN ################## #
def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul=1, bias=True, pre_norm=False, activate = False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

        self.pre_norm = pre_norm
        if pre_norm:
            self.norm = nn.LayerNorm(in_dim, eps=1e-5)
        self.activate = activate
        if self.activate == True:
            self.non_linear = leaky_relu()

    def forward(self, input):
        if hasattr(self, 'pre_norm') and self.pre_norm:
            out = self.norm(input)
            out = F.linear(out, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)
        else:
            out = F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)
        
        if self.activate == True:
            out = self.non_linear(out)
        return out


class Residual(nn.Module):
    def __init__(self,
                 fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)


class StyleVectorizer(nn.Module):
    def __init__(self, dim_in, dim_out, depth, lr_mul = 0.1):
        super().__init__()

        layers = []
        if depth == 1:
            layers.extend([EqualLinear(dim_in, dim_out, lr_mul, pre_norm=False, activate = False)])     
        else:
            for i in range(depth):
                if i == 0:
                    layers.extend([EqualLinear(dim_in, dim_out, lr_mul, pre_norm=False, activate = True)])      
                elif i == depth - 1:
                    layers.extend([EqualLinear(dim_out, dim_out, lr_mul, pre_norm=True, activate = False)])  
                else:
                    layers.extend([Residual(EqualLinear(dim_out, dim_out, lr_mul, pre_norm=True, activate = True))])

        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(dim_out, eps=1e-5)
        
    def forward(self, x):
        return self.norm(self.net(x))
    


class VectorNorm(nn.Module):
    def __init__(self, dim=1, p=2):
        super(VectorNorm, self).__init__()
        self.dim = dim
        self.p = p

    def forward(self, x):
        return F.normalize(x, dim=self.dim, p=self.p)


class VectorSumAs(nn.Module):
    def __init__(self, norm_shape, dim=1, s=1.):
        super(VectorSumAs, self).__init__()
        self.norm_func = nn.BatchNorm1d(norm_shape, eps=1e-05)
        self.dim = dim
        self.s = s

    def forward(self, x):
        x = self.norm_func(x)
        # x = x - x.mean(dim=self.dim)
        return self.s * (x / x.sum(dim=self.dim, keepdims=True))




class MlpBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, out_dim):
        super(MlpBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)



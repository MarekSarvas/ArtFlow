# File for implementing Adaptive Attention Normalization 
# style transfer module for Glow model

import torch
import torch.nn as nn

def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(x):
    size = x.size()
    mean, std = calc_mean_std(x)
    return (x - mean.expand(size)) / std.expand(size)


class AdaAttN(nn.Module):
    def __init__(self, v_dim, qk_dim):
        super(AdaAttN, self).__init__()
        self.f = nn.Conv2d(qk_dim, qk_dim, 1)
        self.g = nn.Conv2d(qk_dim, qk_dim, 1)
        self.h = nn.Conv2d(v_dim, v_dim, 1)

    def forward(self, c_x, s_x, c_1x, s_1x):
        Q = self.f(mean_variance_norm(c_1x))
        Q = Q.flatten(-2, -1).transpose(1, 2)
        K = self.g(mean_variance_norm(s_1x))
        K = K.flatten(-2, -1)
        V = self.h(s_x)
        V = V.flatten(-2, -1).transpose(1, 2)
        A = torch.softmax(torch.bmm(Q, K), -1)
        M = torch.bmm(A, V)
        Var = torch.bmm(A, V ** 2) - M ** 2
        S = torch.sqrt(Var.clamp(min=0))
        M = M.transpose(1, 2).view(c_x.size())
        S = S.transpose(1, 2).view(c_x.size())
        return S * mean_variance_norm(c_x) + M
 

class Transformer(nn.Module):

    def __init__(self, in_planes, key_planes=None, shallow_layer=False):
        super(Transformer, self).__init__()
        self.attn_adain_4_1 = AdaAttN(v_dim=in_planes, qk_dim=key_planes)
        self.attn_adain_5_1 = AdaAttN(v_dim=in_planes,
                                        qk_dim=key_planes + 512 if shallow_layer else key_planes)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))

    def forward(self, content4_1, style4_1, content5_1, style5_1,
                content4_1_key, style4_1_key, content5_1_key, style5_1_key, seed=None):
        return self.merge_conv(self.merge_conv_pad(
            self.attn_adain_4_1(content4_1, style4_1, content4_1_key, style4_1_key, seed=seed) +
            self.upsample5_1(self.attn_adain_5_1(content5_1, style5_1, content5_1_key, style5_1_key, seed=seed))))

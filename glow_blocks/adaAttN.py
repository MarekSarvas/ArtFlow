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

    def __init__(self, in_planes, key_planes, max_sample=256*256):
        super(AdaAttN, self).__init__()
        self.f = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.g = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.max_sample = max_sample

    def forward(self, content, style, content_key, style_key, seed=None):
        F = self.f(mean_variance_norm(content_key))
        G = self.g(mean_variance_norm(style_key))
        H = self.h(style)
        b, _, h_g, w_g = G.size()
        G = G.view(b, -1, w_g * h_g).contiguous()
        if w_g * h_g > self.max_sample:
            if seed is not None:
                torch.manual_seed(seed)
            index = torch.randperm(w_g * h_g).to(content.device)[:self.max_sample]
            G = G[:, :, index]
            style_flat = H.view(b, -1, w_g * h_g)[:, :, index].transpose(1, 2).contiguous()
        else:
            style_flat = H.view(b, -1, w_g * h_g).transpose(1, 2).contiguous()
        b, _, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        S = torch.bmm(F, G)
        # S: b, n_c, n_s
        S = self.sm(S)
        # mean: b, n_c, c
        mean = torch.bmm(S, style_flat)
        # std: b, n_c, c
        std = torch.sqrt(torch.relu(torch.bmm(S, style_flat ** 2) - mean ** 2))
        # mean, std: b, c, h, w
        mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return std * mean_variance_norm(content) + mean
    """

    def forward(self, content, style, content_key, style_key):
        Q = self.f(mean_variance_norm(content_key))
        Q = Q.flatten(-2, -1).transpose(1, 2)
        K = self.g(mean_variance_norm(style_key))
        K = K.flatten(-2, -1)
        V = self.h(style)
        V = V.flatten(-2, -1).transpose(1, 2)
        A = self.sm(torch.bmm(Q, K))
        M = torch.bmm(A, V)
        Var = torch.bmm(A, V ** 2) - M ** 2
        S = torch.sqrt(Var.clamp(min=0))
        M = M.transpose(1, 2).view(content.size())
        S = S.transpose(1, 2).view(content.size())
        return S * mean_variance_norm(content) + M
    """
 

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

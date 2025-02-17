import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from scipy import linalg as la

from glow_blocks.flow_modules import TransitionFlow, Flow
from glow_blocks.style_trans_modules import AdaIN
from glow_blocks.adaAttN import AdaAttN

class Block(nn.Module):
    def __init__(self, in_channel, n_flow, affine=True, conv_lu=True, n_transitions=0):
        super().__init__()

        squeeze_dim = in_channel * 4

        self.flows = nn.ModuleList()
        self.transition_flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu))

        for i in range(n_transitions):
            self.transition_flows.append(TransitionFlow(squeeze_dim, conv_lu=conv_lu))

    def forward(self, input):
        b_size, n_channel, height, width = input.shape
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)

        for flow in self.transition_flows:
            out = flow(out)

        for flow in self.flows:
            out = flow(out)

        return out

    def reverse(self, output, reconstruct=False):
        input = output
        for flow in self.flows[::-1]:
            input = flow.reverse(input)

        for flow in self.transition_flows[::-1]:
            input = flow.reverse(input)

        b_size, n_channel, height, width = input.shape

        unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(b_size, n_channel // 4, height * 2, width * 2)

        return unsqueezed


class Glow(nn.Module):
    def __init__(self, in_channel, n_flow, n_block, affine=True, conv_lu=True, n_trans=0, max_sample=256*256):
        super().__init__()

        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu, n_transitions=n_trans))
            n_channel *= 4
            
        self.blocks.append(Block(n_channel, n_flow, affine=affine, n_transitions=n_trans))
        
        # FIXME: are the dimensions correct? parametrize them..
        self.adaattn = AdaAttN(in_planes= 3 * (4**n_block),
                key_planes=3 * (4**n_block), max_sample=max_sample)
        
    def forward(self, input, forward=True, style=None):
        if forward:
            return self._forward(input, style=style)
        else:
            return self._reverse(input, style=style)

    def _forward(self, input, style=None):
        z = input
        for block in self.blocks:
            z = block(z)
        if style is not None:
            z = self.adaattn(z, style, z, style)
        return z

    def _reverse(self, z, style=None):
        out = z
        if style is not None:
            out = self.adaattn(out, style, out, style)
        for i, block in enumerate(self.blocks[::-1]):
            out = block.reverse(out)
        return out

class GlowIntermediate(nn.Module):
    def __init__(self, in_channel, n_flow, n_block, affine=True, conv_lu=True):
        super().__init__()

        self.blocks = nn.ModuleList()
        n_channel = in_channel
        self.blocks.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu))
        n_channel *= 4
            
        self.blocks.append(Block(n_channel, n_flow, affine=affine))
        
        # FIXME: are the dimensions correct? parametrize them..
        self.adaattn1 = AdaAttN(in_planes= 3 * (4**n_block), key_planes=3 * (4**n_block))
        self.adaattn2 = AdaAttN(in_planes= 3 * (4**n_block), key_planes=3 * (4**(n_block - 1)))
        
    def forward(self, input, forward=True, style=None):
        if forward:
            return self._forward(input, style=style)
        else:
            return self._reverse(input, style=style)

    def _forward(self, input, style=None, return_intermediate=True):
        z = input
        intermediate = []
        for block in self.blocks:
            z = block(z)
            intermediate.append(z)
        if style is not None:
            z = self.adaattn(z, style, z, style)
        return z

    def _reverse(self, z, style=None):
        out = z
        if style is not None:
            out = self.adaattn(out, style, out, style)
        for i, block in enumerate(self.blocks[::-1]):
            out = block.reverse(out)
        return out

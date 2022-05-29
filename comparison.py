import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from pathlib import Path
import matplotlib.pyplot as plt

import time
import numpy as np
import random


def test_transform(img, size):
    transform_list = []
    h, w, _ = np.shape(img)
    if h<w:
        newh = size
        neww = w/h*size
    else:
        neww = size
        newh = h/w*size
    neww = int(neww//4*4)
    newh = int(newh//4*4)
    transform_list.append(transforms.Resize((newh, neww)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str, default='input/content/golden_gate.jpg',
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str, default='input/content',
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str, default='input/style/la_muse.jpg',
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str, default='input/style',
                    help='Directory path to a batch of style images')

# Additional options
parser.add_argument('--size', type=int, default=256,
                    help='New size for the content and style images, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')

# glow parameters
parser.add_argument('--operator', type=str, default='adain',
                    help='style feature transfer operator')
parser.add_argument('--n_flow', default=8, type=int, help='number of flows in each block')# 32
parser.add_argument('--n_trans', default=0, type=int, help='number of transition layers in each block')# 32
parser.add_argument('--n_block', default=2, type=int, help='number of blocks')# 4
parser.add_argument('--max_sample', default=256, type=int, help='maximum adaattn key size')# 32
parser.add_argument('--no_lu', action='store_true', help='use plain convolution instead of LU decomposed version')
parser.add_argument('--affine', default=False, type=bool, help='use affine coupling instead of additive')

# additional arguments
parser.add_argument('--gpu', default=True, type=bool)
parser.add_argument('--pad', default=0, type=int)

parser.add_argument('--decoder', type=str, default='experiments/decoder2.pth.tar')

decoder = ["models/artflow_8_1_2.pth", "models/artflow_16_1_1.pth", "models/glow_adain.pth", "models/glow_wct.pth" ]
net_names = ["content", "style", "arflow_AdaAttN_2", "arflow_AdaAttN_1", "AdaIn", "WCT"]
args = parser.parse_args()

if args.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

from glow_wct import Glow as Glow_wct
from glow_adain import Glow as Glow_adain
from glow_AdaAttN import Glow as Glow_att

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

assert (args.content or args.content_dir)
assert (args.style or args.style_dir)

content_dir = Path(args.content_dir)
content_paths = [f for f in content_dir.glob('*')]
style_dir = Path(args.style_dir)
style_paths = [f for f in style_dir.glob('*')]

# glow
glow_att2 = Glow_att(3, 8, 2, affine=args.affine, conv_lu=not args.no_lu, max_sample=args.max_sample**2, n_trans=1)
glow_att1 = Glow_att(3, 16, 1, affine=args.affine, conv_lu=not args.no_lu, max_sample=args.max_sample**2, n_trans=2)
glow_adain = Glow_adain(3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu)
glow_wct = Glow_wct(3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu)

nets = [glow_att2, glow_att1, glow_adain, glow_wct]

# -----------------------resume training------------------------
if os.path.isfile(args.decoder):
    print("--------loading checkpoint----------")
    for i in range(len(decoder)):
        checkpoint = torch.load(decoder[i], map_location=device)
        nets[i].load_state_dict(checkpoint['state_dict'])
        nets[i].to(device)
        nets[i].eval()
    print("=> loaded checkpoint '{}'".format(args.decoder))
else:
    print("--------no checkpoint found---------")
pad_size = args.pad
# -----------------------start------------------------
fig, axs = plt.subplots(ncols=4+2, nrows=len(content_paths), figsize=(14, 14), constrained_layout=True)

for i in range(len(content_paths)):
    with torch.no_grad():
        content_path = content_paths[i]
        style_path = style_paths[i]

        content = Image.open(str(content_path)).convert('RGB')
        img_transform = test_transform(content, args.size)
        
        content = img_transform(content)
        content = content.to(device).unsqueeze(0)
        
        style = Image.open(str(style_path)).convert('RGB')
        img_transform = test_transform(style, args.size)
        style = img_transform(style)
        style = style.to(device).unsqueeze(0)
        
        axs[i, 0].imshow(content.numpy()[0].transpose((1,2,0)), interpolation='lanczos')
        axs[i, 1].imshow(style.numpy()[0].transpose((1,2,0)), interpolation='lanczos')
        axs[i, 0].axes.xaxis.set_visible(False)
        axs[i, 1].axes.xaxis.set_visible(False)
        axs[i, 0].axes.yaxis.set_visible(False)
        axs[i, 1].axes.yaxis.set_visible(False)
        # padd
        if args.pad != 0:
            content = F.pad(content, (args.pad, args.pad, args.pad, args.pad))

        for j, net in enumerate(nets): 
        # content/style ---> z ---> stylized 
            z_c = net(content, forward=True)
            z_s = net(style, forward=True)
            output = net(z_c, forward=False, style=z_s)

            if args.pad != 0:
                output = output[:, :, args.pad:-args.pad, args.pad:-args.pad]

            output = output.cpu()[0]
            print(output.shape)
            axs[i, j+2].imshow(output.numpy().transpose((1,2,0)), interpolation='lanczos')
            axs[i, j+2].axes.xaxis.set_visible(False)
            axs[i, j+2].axes.yaxis.set_visible(False)

            output_name = output_dir / '{}_{:s}_stylized_{:s}{:s}'.format(net_names[j], content_path.stem, style_path.stem, args.save_ext)
            print(output_name)
            save_image(output, str(output_name))
for ax, col in zip(axs[0], net_names):
    ax.set_title(col)
#plt.tight_layout()
plt.subplots_adjust(hspace=0.001)
plt.savefig("comparison.pdf")
plt.show()
            

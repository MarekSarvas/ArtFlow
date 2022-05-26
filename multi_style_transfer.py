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
import torch.utils.data as data
from sampler import InfiniteSamplerWrapper

import time
import numpy as np
import random

import matplotlib.pyplot as plt


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
    
def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform, size, train):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = transform
        self.size = size
        self.train = train

    def __getitem__(self, index):
        path = self.paths[index]
        if self.train:
            img = Image.open(os.path.join(self.root, path)).convert('RGB')
            img = self.transform()(img)
        else:
            img = Image.open(os.path.join(self.root, path)).convert('RGB')
            img = self.transform(img, self.size)(img)

        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def style_interpolation(z1, z2, n):
    print("Style 1: ", z1.shape)
    print("Style 2: ", z2.shape)
    step = 1/n
    z_new = [z1]
    titles = ["Style A", "1 : 0"]
    p = 0+step
    for _ in range(n-2):
        z_tmp = p*z2 + (1-p)*z1 
        z_new.append(z_tmp)
        p = p+step
        titles.append("{:.1f} : {:.1f}".format(1-p, p))
    z_new.append(z2)
    titles.append("0 : 1")
    titles.append("Style B")

    return z_new, titles


def plot_interpolation(imgs, save_path, titles, show=True):
    num_imgs = len(imgs[1])+2
    fig, axs = plt.subplots(1, num_imgs, figsize=(16,4))

    for i in range(num_imgs):
        if i == 0:
            axs[i].imshow(imgs[0][0].numpy().transpose((1,2,0)), interpolation='nearest')
            axs[i].set_title(titles[i])
        elif i == num_imgs-1:
            axs[i].imshow(imgs[2][0].numpy().transpose((1,2,0)), interpolation='spline16')
            axs[i].set_title(titles[i])
        else:
            axs[i].imshow(imgs[1][i-1][0].numpy().transpose((1,2,0)), interpolation='lanczos')
            axs[i].set_title(titles[i])
        axs[i].axes.xaxis.set_visible(False)
        axs[i].axes.yaxis.set_visible(False)
    fig.suptitle('Interpolation between 2 styles') 
    plt.tight_layout() 
    plt.savefig(save_path)
    if show:
        plt.show()
    pass


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
parser.add_argument('--decoder', type=str, default='experiments/decoder2.pth.tar')

# Additional options
parser.add_argument('--size', type=int, default=256,
                    help='New size for the content and style images, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.pdf',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')

# glow parameters
parser.add_argument('--operator', type=str, default='adain',
                    help='style feature transfer operator')
parser.add_argument('--n_flow', default=8, type=int, help='number of flows in each block')# 32
parser.add_argument('--n_block', default=2, type=int, help='number of blocks')# 4
parser.add_argument('--no_lu', action='store_true', help='use plain convolution instead of LU decomposed version')
parser.add_argument('--affine', default=False, type=bool, help='use affine coupling instead of additive')

# additional args
parser.add_argument("--gpu", default=True, type=bool)
parser.add_argument("--batch_size", default=1, type=int)


args = parser.parse_args()
if args.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

if args.operator == 'wct':
    from glow_wct import Glow
elif args.operator == 'adain':
    from glow_adain import Glow
elif args.operator == 'decorator':
    from glow_decorator import Glow
elif args.operator == 'att':
    from glow_AdaAttN import Glow
else:
    raise('Not implemented operator', args.operator)

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

assert (args.content or args.content_dir)
assert (args.style or args.style_dir)

content_dir = Path(args.content_dir)
content_paths = [f for f in content_dir.glob('*')]
style_dir = Path(args.style_dir)
style_paths = [f for f in style_dir.glob('*')]

# glow
glow = Glow(3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu)

# -----------------------resume training------------------------
if os.path.isfile(args.decoder):
    print("--------Load trained model----------")
    checkpoint = torch.load(args.decoder, map_location=device)
    args.start_iter = checkpoint['iter']
    glow.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}'".format(args.decoder))
else:
    print("--------No checkpoints---------")
glow = glow.to(device)

glow.eval()

# -----------------------start------------------------
train = False
if train:
    content_tf = train_transform
    style_tf = train_transform
else:
    content_tf = test_transform
    style_tf = test_transform

content_tf = test_transform
style_tf = train_transform

content_dataset = FlatFolderDataset(args.content_dir, content_tf, args.size, False)
style_dataset = FlatFolderDataset(args.style_dir, style_tf, args.size, True)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset)))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset)))


#breakpoint()
prev=0
for i in range(20):
    i += prev
    with torch.no_grad():
        content = next(content_iter).to(device)
        style1 = next(style_iter).to(device)
        style2 = next(style_iter).to(device)

        z_c = glow(content, forward=True)
        z_s1 = glow(style1, forward=True)
        z_s2 = glow(style2, forward=True)

        z_combined, titles = style_interpolation(z_s1, z_s2, 5) 
        outputs = []
        for j in range(len(z_combined)):
            output = glow(z_c, forward=False, style=z_combined[j])
            output = output.cpu()
            outputs.append(output)
        
        output_name = output_dir / "interpolation_{}_8_2_model{:s}".format(i, args.save_ext)
        output_c = output_dir / "content{}.{}".format(i, args.save_ext)
        print(output_name)
       # breakpoint()
        plot_interpolation([style1, outputs, style2], output_name, titles, show=False)
        save_image(content, str(output_c))

exit(69)







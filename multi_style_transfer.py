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
    
def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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
parser.add_argument('--save_ext', default='.jpg',
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

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    checkpoint = torch.load(args.decoder)
    args.start_iter = checkpoint['iter']
    glow.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}'".format(args.decoder))
else:
    print("--------Wrong path to the model---------")
    exit(1)
glow = glow.to(device)

glow.eval()

# -----------------------start------------------------
content_tf = train_transform()
style_tf = train_transform()

content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

optimizer = torch.optim.Adam(glow.module.parameters(), lr=args.lr)
if args.resume:
    if os.path.isfile(args.resume):
        optimizer.load_state_dict(checkpoint['optimizer'])

log_c = []
log_s = []
log_mse = []
Time = time.time()
# -----------------------training------------------------
for i in range(args.start_iter, args.max_iter):
    adjust_learning_rate(optimizer, iteration_count=i)
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)

    # glow forward: real -> z_real, style -> z_style
    if i == args.start_iter:
        with torch.no_grad():
            _ = glow.module(content_images, forward=True)
            continue

    # (log_p, logdet, z_outs) = glow()
    z_c = glow(content_images, forward=True)
    z_s = glow(style_images, forward=True)
    # reverse 
    stylized = glow(z_c, forward=False, style=z_s)

    loss_c, loss_s = encoder(content_images, style_images, stylized)
    loss_c = loss_c.mean()
    loss_s = loss_s.mean()
    loss_mse = mseloss(content_images, stylized)
    loss_style = args.content_weight*loss_c + args.style_weight*loss_s + args.mse_weight*loss_mse

    # optimizer update
    optimizer.zero_grad()
    loss_style.backward()
    nn.utils.clip_grad_norm(glow.module.parameters(), 5)
    optimizer.step()
    
    # update loss log
    log_c.append(loss_c.item())
    log_s.append(loss_s.item())
    log_mse.append(loss_mse.item())

    # save image
    if i % args.print_interval == 0:
        with torch.no_grad():
            # stylized ---> z ---> content
            z_stylized = glow(stylized, forward=True)
            real = glow(z_stylized, forward=False, style=z_c)
            
            # pick another content
            another_content = next(content_iter).to(device)

            # stylized ---> z ---> another real
            z_ac = glow(another_content, forward=True)
            another_real = glow(z_stylized, forward=False, style=z_ac)

        output_name = os.path.join(args.save_dir, "%06d.jpg" % i)
        output_images = torch.cat((content_images.cpu(), style_images.cpu(), stylized.cpu(), 
                                    real.cpu(), another_content.cpu(), another_real.cpu()), 
                                  0)
        save_image(output_images, output_name, nrow=args.batch_size)
        
        print("iter %d   time/iter: %.2f   loss_c: %.3f   loss_s: %.3f   loss_mse: %.3f" % (i, 
                                                                      (time.time()-Time)/args.print_interval, 
                                                                      np.mean(np.array(log_c)), np.mean(np.array(log_s)),
                                                                      np.mean(np.array(log_mse))
                                                                       ))
        log_c = []
        log_s = []
        Time = time.time()

        
    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = glow.module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))

        state = {'iter': i, 'state_dict': state_dict, 'optimizer': optimizer.state_dict()}
        torch.save(state, args.resume)


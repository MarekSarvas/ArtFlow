import torch
import json
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import PIL
import os
import time
import argparse
"""
    python3 eval_dataset.py -d dataset_root_path -o  json file
"""


models = ['alexnet', 'vgg11', 'vgg13', 'vgg16',
          'vgg19', 'vgg19_bn', 'densenet161', 'inception_v3', 'resnet152', ]
imsize = 299
use_cuda = True

preprocess = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor(),
])


def image_loader(path, device):
    image = PIL.Image.open(path)
    image = preprocess(image).unsqueeze(0)
    return image.to(device, torch.float)


def imshow(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    plt.show()


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std


def model_loader(name):

    return nn.Sequential(
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        torch.hub.load('pytorch/vision:v0.6.0', name, pretrained=True)
    ).to(device).eval()


def imshow_batch(input, title):
    # torch.Tensor => numpy
    input = input.numpy().transpose((1, 2, 0))
    # display images
    plt.imshow(input)
    plt.title(title)
    plt.savefig('Batch')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample app')
    parser.add_argument("-d", "--dataset", action="store",
                        dest="dataset")
    parser.add_argument("-o", "--output", action="store", dest="output")
    arguments = parser.parse_args()
    if(arguments.dataset == None or arguments.output == None):
        print("Missing arguments")
        exit(1)

    data_dir = arguments.dataset
    output_file = arguments.output
    output_dict = {}

    with open('./imagenet.json') as f:
        imagenet_labels = json.load(f)

    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    val_datasets = datasets.ImageFolder(os.path.join(data_dir), preprocess)
    val_dataloader = torch.utils.data.DataLoader(
        val_datasets, batch_size=16, shuffle=True, num_workers=4)
    print('Validation dataset size:', len(val_datasets))
    class_names = val_datasets.classes
    print('The number of classes:', len(class_names))

    for model_name in models:
        print(model_name)
        model = model_loader(model_name)
        criterion = nn.CrossEntropyLoss()
        model.eval()
        start_time = time.time()

        with torch.no_grad():
            running_loss = 0.
            running_corrects = 0

            for i, (inputs, basic_labels) in enumerate(val_dataloader):
                inputs = inputs.to(device)
                labels = torch.zeros_like(basic_labels).to(device)
                for j in range(labels.shape[0]):
                    labels[j] = int(class_names[basic_labels[j]])
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                """
                nechat pre obrazky do docu
                if i == 0:
                    print('[Prediction Result Examples]')
                    images = torchvision.utils.make_grid(inputs[:4])
                    imshow_batch(images.cpu(), title='predicted labels:' + str([int(x) for x in preds[:4]]))
                    print('Real labels >>>>>>>>>>>>>>>>>>>>>>')
                    for j, imagenet_index in enumerate(labels[:4]):
                        label = imagenet_labels[imagenet_index]
                        print(f'Image #{j + 1}: {label} ({imagenet_index})')
                    images = torchvision.utils.make_grid(inputs[4:8])
                    imshow_batch(images.cpu(), title='predicted labels:' + str([int(x) for x in preds[4:8]]))
                    print('Real labels >>>>>>>>>>>>>>>>>>>>>>')
                    for j, imagenet_index in enumerate(labels[4:8]):
                        label = imagenet_labels[imagenet_index]
                        print(f'Image #{j + 5}: {label} ({imagenet_index})')
                """
            epoch_loss = running_loss / len(val_datasets)
            epoch_acc = running_corrects / len(val_datasets) * 100.
            print('[Validation] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(
                epoch_loss, epoch_acc, time.time() - start_time))
            output_dict[model_name] = {"epoch_loss": float(epoch_loss),
                                       "epoch_acc": float(epoch_acc),
                                       "duration": time.time() - start_time}

    with open(output_file, "w") as fp:
        json.dump(output_dict, fp)

import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np

import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms, datasets

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_backgrounds():
    backgrounds = []
    for file in os.listdir("./images/train"):
        if file.endswith('.jpg'):
            backgrounds.append(plt.imread(os.path.join("./images/train",file)))
    return np.array(backgrounds)
backgrounds = get_backgrounds()


def compose_image(image):
    image = (image > 0).astype(np.float32)
    image = image.reshape([28,28])*255.0
    
    image = np.stack([image,image,image],axis=2)
    
    background = np.random.choice(backgrounds)
    w,h,_ = background.shape
    dw, dh,_ = image.shape
    x = np.random.randint(0,w-dw)
    y = np.random.randint(0,h-dh)
    
    temp = background[x:x+dw, y:y+dh]
    return np.abs(temp-image).astype(np.uint8)


class MNISTM(Dataset):
            
    def __init__(self, train=True,transform=None):
        if train:
            self.data = datasets.MNIST(root='.data/mnist',train=True, download=True)
        else:
            self.data = datasets.MNIST(root='.data/mnist',train=False, download=True)
        self.backgrounds = get_backgrounds()
        self.transform = transform
        self.images = []
        self.targets = []
        for index in range(len(self.data)):
            image = np.array(self.data.__getitem__(index)[0])
            target = self.data.__getitem__(index)[1]
            image = compose_image(image)
            if self.transform is not None:
                image = self.transform(image)
            self.images.append(image)
            self.targets.append(target)
        
    def __getitem__(self,index):
        
        #image = Image.fromarray(image.squeeze(), mode="RGB")
        image = self.images[index]
        target = self.targets[index]
        
        return image, target
        
    def __len__(self):
        return len(self.data)
    
kwargs =  {'num_workers': 2, 'pin_memory': True}

def get_mnistm_loaders(data_aug = False, batch_size=128,test_batch_size=1000):
    if data_aug:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(28),
            transforms.RandomCrop(28,padding=4),
            transforms.ToTensor()
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(28),
            transforms.ToTensor()
        ])
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(28),
        transforms.ToTensor()
    ])

    train_loader = DataLoader(
        MNISTM(train=True,transform=train_transform),batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    train_eval_loader = DataLoader(
        MNISTM(train=True, transform=test_transform),batch_size=test_batch_size, shuffle=False, drop_last=True, **kwargs)
    test_loader = DataLoader(
        MNISTM(train=False,transform=test_transform),batch_size=test_batch_size, shuffle=False, drop_last=True, **kwargs)
    return train_loader, train_eval_loader, test_loader


def get_mnist_loaders(data_aug = False, batch_size=128,test_batch_size=1000):
    if data_aug:
        train_transform = transforms.Compose(
            [transforms.Resize(28),
            transforms.RandomCrop(28,padding=4),
            transforms.Grayscale(3),
            transforms.ToTensor()
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(28),
            transforms.Grayscale(3),
            transforms.ToTensor()
        ])
    test_transform = transforms.Compose([
        transforms.Resize(28),
        transforms.Grayscale(3),
        transforms.ToTensor()
    ])
  

    train_loader = DataLoader(
        datasets.MNIST(root='.data/mnist',train=True, download=True,transform=train_transform),batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    train_eval_loader = DataLoader(
        datasets.MNIST(root='.data/mnist',train=True, download=True, transform=test_transform),batch_size=test_batch_size, shuffle=False, drop_last=True, **kwargs)
    test_loader = DataLoader(
        datasets.MNIST(root='.data/mnist',train=False, download=True, transform=test_transform),batch_size=test_batch_size, shuffle=False, drop_last=True, **kwargs)
    return train_loader, train_eval_loader, test_loader
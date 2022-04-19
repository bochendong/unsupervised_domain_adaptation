import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

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
    def __getitem__(self,index):
        image = np.array(self.data.__getitem__(index)[0])
        target = self.data.__getitem__(index)[1]
        image = compose_image(image)
        #image = Image.fromarray(image.squeeze(), mode="RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target
        
    def __len__(self):
        return len(self.data)
    
    
def get_mnistm_loaders(data_aug = False, batch_size=128,test_batch_size=1000):
    if data_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(28,padding=4),
            transforms.ToTensor()
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor()
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    kwargs = {}
    train_loader = DataLoader(
        MNISTM(train=True,transform=train_transform),batch_size=batch_size, shuffle=True, drop_last=True)
    train_eval_loader = DataLoader(
        MNISTM(train=True, transform=test_transform),batch_size=test_batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(
        MNISTM(train=False,transform=test_transform),batch_size=test_batch_size, shuffle=False, drop_last=True)
    return train_loader, train_eval_loader, test_loader


def get_mnist_loaders(data_aug = False, batch_size=128,test_batch_size=1000):
    if data_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(28,padding=4),
            transforms.Grayscale(3),
            transforms.ToTensor()
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Grayscale(3),
            transforms.ToTensor()
        ])
    test_transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.ToTensor()
    ])
    kwargs = {}

    train_loader = DataLoader(
        datasets.MNIST(root='.data/mnist',train=True, download=True,transform=train_transform),batch_size=batch_size, shuffle=True, drop_last=True)
    train_eval_loader = DataLoader(
        datasets.MNIST(root='.data/mnist',train=True, download=True, transform=test_transform),batch_size=test_batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(
        datasets.MNIST(root='.data/mnist',train=False, download=True, transform=test_transform),batch_size=test_batch_size, shuffle=False, drop_last=True)
    return train_loader, train_eval_loader, test_loader
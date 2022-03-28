import torch
from torchvision import datasets, transforms

def load_data():
    image_size = 32
    batch_size = 128
    transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])

    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)

    kwargs = {}
    if (torch.cuda.is_available()):
        kwargs = {'num_workers': 2, 'pin_memory': True}

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                shuffle=True, **kwargs)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                                shuffle=False, **kwargs)


    return trainloader, testloader

import torch
import torchvision
import torchvision.transforms as transforms


class CIFAR10DataLoader:
    def __init__(self):
        self.transform = {'training': transforms.Compose([
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]), 'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])}
        self.data = {'training': torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                              transform=self.transform['training']),
                     'test': torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                                          transform=self.transform['test'])}

    def read_data_set(self, data_set: str):
        return self.data[data_set]

    def get_data_loader(self, data_set: str, batch_size: int):
        return torch.utils.data.DataLoader(self.data[data_set], batch_size=batch_size)

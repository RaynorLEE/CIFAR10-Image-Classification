import torch.nn as nn
import torch.nn.functional as F


class DnCNN(nn.Module):
    def __init__(self):
        super(DnCNN, self).__init__()
        kernel_size = 3
        features = 16
        self.conv = nn.Conv2d(3, 16, 3)
        layers = []
        for depth in range(3):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features * 2, kernel_size=kernel_size))
            features *= 2
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        for depth in range(4):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 4096)
        self.fc2 = nn.Linear(4096, 2560)
        self.fc3 = nn.Linear(2560, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.layers(x)
        x = self.pool(x)
        x = x.view(x.size(0), 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x

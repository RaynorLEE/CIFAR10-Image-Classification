import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        #   input image: 3 x 32 x 32 signal
        #   Convolution Layer 1: 3 x 32 x 32 -> 6 x 28 x 28 with 5x5 convolution blocks
        self.conv1 = nn.Conv2d(3, 6, 5)
        #   Pooling: 6 x 28 x 28 -> 6 x 14 x 14 with step length = 2
        self.pool = nn.MaxPool2d(2, 2)
        #   Convolution Layer 2: 6 x 14 x 14 -> 16 x 10 x 10 with 5x5 convolution blocks
        self.conv2 = nn.Conv2d(6, 16, 5)
        #   Full Connection Layer 1: input = 16 x 5 x 5, output = 120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        #   Full Connection Layer 2: input = 120, output = 84
        self.fc2 = nn.Linear(120, 84)
        #   Full Connection Layer 3: input = 84, output = 10
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #   input image: 32 x 32 x 3 signal
        self.conv1 = nn.Conv2d(3, 16, 3)  # 3 x 32 x 32 -> 16 x 30 x 30
        self.conv2 = nn.Conv2d(16, 32, 3)  # 16 x 30 x 30 -> 32 x 28 x 28
        self.conv3 = nn.Conv2d(32, 48, 3)   # 32 x 28 x 28 -> 48 x 26 x 26
        self.conv4 = nn.Conv2d(48, 64, 3)   # 48 x 26 x 26 -> 64 x 24 x 24
        self.conv5 = nn.Conv2d(64, 64, 3)   # 64 x 24 x 24 -> 64 x 22 x 22
        self.conv6 = nn.Conv2d(64, 64, 3)   # 64 x 22 x 22 -> 64 x 20 x 20
        self.conv7 = nn.Conv2d(64, 64, 3)   # 64 x 20 x 20 -> 64 x 18 x 18
        self.pool = nn.MaxPool2d(2, 2)      # 64 x 18 x 18 -> 64 x 9 x 9
        self.fc1 = nn.Linear(64 * 9 * 9, 3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 128)
        self.fc6 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(F.relu(self.conv7(x)))
        x = x.view(-1, 64 * 9 * 9)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x

import torch.nn as nn
import torch.nn.functional as F


class WideResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate, stride):
        super(WideResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=3, stride=stride, padding=1, bias=True)
        )
        self.shortcut = nn.Sequential()
        if stride > 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=1, stride=stride, bias=True)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out


def construct_layer(in_channels, out_channels, depth, dropout_rate, stride):
    layers = [WideResBlock(in_channels, out_channels, dropout_rate, stride), ]
    for i in range(depth - 1):
        layers.append(WideResBlock(out_channels, out_channels, dropout_rate, 1))
    return nn.Sequential(*layers)


class WideResNet(nn.Module):
    def __init__(self):
        super(WideResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = construct_layer(in_channels=16, out_channels=160, depth=4, dropout_rate=0.3, stride=1)
        self.conv3 = construct_layer(in_channels=160, out_channels=320, depth=4, dropout_rate=0.3, stride=2)
        self.conv4 = construct_layer(in_channels=320, out_channels=640, depth=4, dropout_rate=0.3, stride=2)
        self.bn = nn.BatchNorm2d(640, momentum=0.9)
        self.pool = nn.AvgPool2d(8, 8)
        self.fc = nn.Linear(640, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.bn(out)
        out = F.relu(out)
        out = nn.pool(out)
        out = out.view(out.size(0), 640)
        out = self.fc(out)
        return out

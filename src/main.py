import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from model import *

#   Load and Normalize cifar10 training and test set
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
training_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
training_set_loader = torch.utils.data.DataLoader(training_set, batch_size=4)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_set_loader = torch.utils.data.DataLoader(test_set, batch_size=4)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#   method to display a image
def img_show(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


#   show some data sample
data_iter = iter(training_set_loader)
images, labels = data_iter.next()
img_show(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

#   Initialize Convolutional Neural Network
cnn = CNN()

#   Define Loss function
criterion = nn.CrossEntropyLoss()
#   Define Optimizer
print(cnn)
print(cnn.parameters())
optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

#   Train network model
for epoch in range(2):
    for data in training_set_loader:
        print(data)
        break
    pass



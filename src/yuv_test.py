import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import cv2
from PIL import Image
from src.model import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#   Load and Normalize cifar10 training and test set
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
training_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
training_set_loader = torch.utils.data.DataLoader(training_set, batch_size=4)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
test_set_loader = torch.utils.data.DataLoader(test_set, batch_size=4)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print(training_set)
training_set_yuv = []
for data in training_set:
    img = transform(data[0].convert('YCbCr'))
    label = data[1]
    training_set_yuv.append((img, label))

test_set_yuv = []
for data in test_set:
    img = transform(data[0].convert('YCbCr'))
    label = data[1]
    test_set_yuv.append((img, label))

training_set_yuv_loader = torch.utils.data.DataLoader(training_set_yuv, batch_size=32)
test_set_yuv_loader = torch.utils.data.DataLoader(test_set_yuv, batch_size=32)

#   Initialize Convolution Neural Network
cnn = ResNet()
print(cnn)
print(cnn.parameters())
loss_func = nn.CrossEntropyLoss()
#optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(cnn.parameters(), lr=1e-4, weight_decay=0.001)

if torch.cuda.is_available():
    cnn = cnn.cuda()

plt_x = []
plt_train_y = []
plt_test_y = []
for epoch in range(20):
    running_loss = 0.0
    mini_batch = 0
    for data in training_set_yuv_loader:
        mini_batch += 1
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        prediction = cnn(images)
        loss = loss_func(prediction, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if mini_batch % 500 == 499:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, mini_batch, running_loss / 2000))
            running_loss = 0.0
    print('Epoch = ', epoch + 1)
    plt_x.append(epoch + 1)
    # Train accuracy:
    correct = 0
    total = 0
    with torch.no_grad():
        for data in training_set_yuv_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = cnn(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    plt_train_y.append(accuracy)
    print('Training Accuracy of the model: %.2f %%' % (100 * correct / total))
    # Test Accuracy:
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_set_yuv_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = cnn(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    plt_test_y.append(accuracy)
    print('Test Accuracy of the model: %.2f %%' % (100 * correct / total))
torch.save(cnn, './model/ResNet_512_YUV')
print('Finished Training')

print(plt_x)
print(plt_train_y)
print(plt_test_y)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(plt_x, plt_train_y)
plt.plot(plt_x, plt_test_y)
plt.show()






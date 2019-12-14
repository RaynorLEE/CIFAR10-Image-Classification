import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import csv
from src.model import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#   Load and Normalize cifar10 training and test set
train_transform = transforms.Compose([
    transforms.RandomCrop(32, 2),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
training_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
training_set_loader = torch.utils.data.DataLoader(training_set, batch_size=32)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=train_transform)
test_set_loader = torch.utils.data.DataLoader(test_set, batch_size=32)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#   Initialize Convolution Neural Network
cnn = ResNet()
#   cnn = torch.load('./model/ResNet_512_RGB_no_BN')
#   cnn.eval()
print(cnn)
print(cnn.parameters())

#   Define Loss function
loss_func = nn.CrossEntropyLoss()
#   Define Optimizer
#   optimizer = optim.SGD(cnn.parameters(), lr=0.0001, momentum=0.9)
optimizer = optim.Adam(cnn.parameters(), lr=1e-4, weight_decay=0.001)

if torch.cuda.is_available():
    cnn = cnn.cuda()

#   Train network model
plt_x = []
plt_train_y = []
plt_test_y = []
for epoch in range(100):
    running_loss = 0.0
    mini_batch = 0
    for data in training_set_loader:
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
        if mini_batch % 200 == 199:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, mini_batch, running_loss / 2000))
            running_loss = 0.0
    print('Epoch = ', epoch + 1)
    plt_x.append(epoch + 1)
    # Train accuracy:
    correct = 0
    total = 0
    with torch.no_grad():
        for data in training_set_loader:
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
        for data in test_set_loader:
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
    if epoch % 5 == 4:
        torch.save(cnn, './model/ResNet_512_RGB_no_BN_epo_{}'.format(epoch+1))
print('Finished Training\n\n')

print(plt_x)
print(plt_train_y)
print(plt_test_y)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(plt_x, plt_train_y)
plt.plot(plt_x, plt_test_y)
plt.show()

y_test = np.load("./test/y_test.npy")
y_test_normalized = []
for data in y_test:
    data = test_transform(data)
    y_test_normalized.append(data)
y_loader = torch.utils.data.DataLoader(y_test_normalized, batch_size=32)

y_result = []
with torch.no_grad():
    index = 0
    for data in y_loader:
        data = data.to(device)
        outputs = cnn(data)
        _, predicted = torch.max(outputs.data, 1)
        for i in range(len(predicted)):
            y_result.append([index+i, int(predicted[i])])
        index += 32

with open('./result/test.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Index', 'Category'])
    writer.writerows(y_result)


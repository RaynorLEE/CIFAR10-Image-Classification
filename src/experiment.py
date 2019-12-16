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
    transforms.RandomCrop(32, 4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
training_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
training_set_loader = torch.utils.data.DataLoader(training_set, batch_size=256)
print(type(training_set_loader))
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=train_transform)
test_set_loader = torch.utils.data.DataLoader(test_set, batch_size=256)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#   Initialize Convolution Neural Network
cnn = ResNet()
# cnn = torch.load('./model/ResNet_512_RGB_BN_epo_40')
# cnn.eval()
print(cnn)
print(cnn.parameters())

#   Define Loss function
loss_func = nn.CrossEntropyLoss()
#   Define Optimizer
#   optimizer = optim.SGD(cnn.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
optimizer = optim.Adam(cnn.parameters(), lr=1e-4, weight_decay=1e-4)

if torch.cuda.is_available():
    cnn = cnn.cuda()

#   Train network model
plt_x = []
plt_train_y = []
plt_test_y = []
best_test_accuracy = 0
best_accuracy_epo = 0
for epoch in range(300):
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
        if mini_batch % 195 == 194:
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
    training_set_accuracy = 100 * correct / total
    plt_train_y.append(training_set_accuracy)
    print('Training Accuracy of the model: %.2f %%' % training_set_accuracy)
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
    test_set_accuracy = 100 * correct / total
    plt_test_y.append(test_set_accuracy)
    print('Test Accuracy of the model: %.2f %%' % (100 * correct / total))
    if epoch % 5 == 4:
        torch.save(cnn, './model/ResNet_512_RGB_BN_epo_{}'.format(epoch+1+40))
    if test_set_accuracy > best_test_accuracy:
        if test_set_accuracy > 85:
            torch.save(cnn, './model/ResNet-18_best_accuracy'.format(epoch + 1))
        best_accuracy_epo = epoch + 1
        best_test_accuracy = test_set_accuracy
    print('Best record: epoch = %d, Accuracy = %.2f %%' % (best_accuracy_epo, best_test_accuracy))
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
y_loader = torch.utils.data.DataLoader(y_test_normalized, batch_size=128)

y_result = []
with torch.no_grad():
    index = 0
    for data in y_loader:
        data = data.to(device)
        outputs = cnn(data)
        _, predicted = torch.max(outputs.data, 1)
        for i in range(len(predicted)):
            y_result.append([index+i, int(predicted[i])])
        index += 128

with open('./result/test.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Index', 'Category'])
    writer.writerows(y_result)

print("Test report generated.")
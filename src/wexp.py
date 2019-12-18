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
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=train_transform)
batch_size = 128
training_set_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size)
test_set_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#   Initialize Convolution Neural Network
#   cnn = WideResNet()
cnn = torch.load('./model/Wide_ResNet_best_accuracy')
cnn.eval()
print(cnn)
print(cnn.parameters())

#   Define Loss function
loss_func = nn.CrossEntropyLoss()
#   Define Optimizer
optimizer = optim.SGD(cnn.parameters(), lr=0.00016, momentum=0.9, weight_decay=1e-3)
#   optimizer = optim.Adam(cnn.parameters(), lr=1e-4, weight_decay=1e-4)

if torch.cuda.is_available():
    cnn = cnn.cuda()

#   Train network model
#   Note: only TRAINING set will be used to train the network
plt_x = []
plt_train_y = []
plt_test_y = []
best_test_accuracy = 0
best_accuracy_epo = 0
for epoch in range(30):
    print('Epoch = ', epoch + 1)
    running_loss = 0.0
    mini_batch = 0
    #   Increase learning rate and weight_decay correspondingly
    #   Stage 1: epoch 1 - 50: lr = 1e-2, weight_decay = 5e-3
    #   Stage 2: epoch 51 - 70: lr = 1e-3, weight_decay = 5e-3
    #   Stage 3: epoch 71 up: lr = 1e-3, weight_decay = 1e-2
    # if epoch == 50:
    #     optimizer = optim.SGD(cnn.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-4)
    # if epoch == 80:
    #     optimizer = optim.SGD(cnn.parameters(), lr=0.004, momentum=0.9, weight_decay=5e-4)
    # if epoch == 120:
    #     optimizer = optim.SGD(cnn.parameters(), lr=0.0008, momentum=0.9, weight_decay=5e-4)
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
        #   Output Cross-entropy Loss twice per epoch
        check_per_batch = int(50000 / batch_size / 2)
        if mini_batch % check_per_batch == check_per_batch - 1:
            print('[%d, %5d] loss: %.5f' % (epoch + 1, mini_batch + 1, running_loss / check_per_batch))
            running_loss = 0.0
    #   Accuracy Evaluation
    plt_x.append(epoch + 1)
    #   Training accuracy:
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
    print('Training Accuracy of the model: %.4f %%' % training_set_accuracy)
    #   Testing Accuracy:
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
    print('Test Accuracy of the model: %.4f %%' % (100 * correct / total))
    if epoch % 5 == 4:
        torch.save(cnn, './model/Wide_ResNet_batch-size={}_epo_{}'.format(batch_size, epoch + 1))
    if test_set_accuracy > best_test_accuracy:
        if test_set_accuracy > 85:
            torch.save(cnn, './model/Wide_ResNet_best_accuracy_batch-size={}_epo={}'.format(batch_size, epoch + 1))
        best_accuracy_epo = epoch + 1
        best_test_accuracy = test_set_accuracy
    print('Best record: epoch = %d, Accuracy = %.4f %%\n' % (best_accuracy_epo, best_test_accuracy))
print('Finished Training\n\n')

#   Show accuracy result
print(plt_x)
print(plt_train_y)
print(plt_test_y)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(plt_x, plt_train_y, label='Training Accuracy')
plt.plot(plt_x, plt_test_y, label='Testing Accuracy')
plt.legend()
plt.title('Training and Test Accuracy per Epoch')
plt.show()

#   Generate Test Report
print('Loading Best Network model...')
cnn = torch.load('./model/Wide_ResNet_best_accuracy_batch-size={}_epo={}'.format(batch_size, best_accuracy_epo + 1))
cnn.eval()
if torch.cuda.is_available():
    cnn = cnn.cuda()
print('Loading test set...')
y_test = np.load("./test/y_test.npy")
y_test_normalized = []
for data in y_test:
    data = test_transform(data)
    y_test_normalized.append(data)
y_loader = torch.utils.data.DataLoader(y_test_normalized, batch_size=128)

print('Start testing process...')
y_result = []
with torch.no_grad():
    index = 0
    for data in y_loader:
        data = data.to(device)
        outputs = cnn(data)
        _, predicted = torch.max(outputs.data, 1)
        for i in range(len(predicted)):
            y_result.append([index + i, int(predicted[i])])
        index += 128

with open('./result/y_test_result.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Index', 'Category'])
    writer.writerows(y_result)

print("Test report generated.")

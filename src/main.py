import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from src.model import *

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

#   Initialize Convolution Neural Network
cnn = LeNet()

#   Define Loss function
loss_func = nn.CrossEntropyLoss()
#   Define Optimizer
print(cnn)
print(cnn.parameters())
optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

#   Train network model
for epoch in range(2):
    running_loss = 0.0
    mini_batch = 0
    for data in training_set_loader:
        mini_batch += 1
        images, labels = data
        prediction = cnn(images)
        loss = loss_func(prediction, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if mini_batch % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, mini_batch, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')

dataiter = iter(test_set_loader)
images, labels = dataiter.next()

# print images
img_show(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

outputs = cnn(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

#   Testing
correct = 0
total = 0
with torch.no_grad():
    for data in test_set_loader:
        images, labels = data
        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Testing Accuracy of the model: %d %%' % (100 * correct / total))


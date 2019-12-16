import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
import csv

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

print('Loading test images...')
y_test = np.load("./test/y_test.npy")
y_test_normalized = []
for data in y_test:
    data = test_transform(data)
    y_test_normalized.append(data)
y_loader = torch.utils.data.DataLoader(y_test_normalized, batch_size=256)

print('Loading network model...')
cnn = torch.load('./model/ResNet-18_best_accuracy')
cnn.eval()

if torch.cuda.is_available():
    cnn = cnn.cuda()

print('Start testing...')
y_result = []
batch_size = 256
with torch.no_grad():
    index = 0
    print('Current progress: %.3f%%' % (100 * index * batch_size / 12000))
    for data in y_loader:
        data = data.to(device)
        outputs = cnn(data)
        _, predicted = torch.max(outputs.data, 1)
        for i in range(len(predicted)):
            y_result.append([index+i, int(predicted[i])])
        index += batch_size

with open('./result/y_test_result.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Index', 'Category'])
    writer.writerows(y_result)

print('Test result form generated.')
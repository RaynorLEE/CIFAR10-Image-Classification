import torch.optim as optim
from src.model import *
from src.data import *
from src.graph import draw_accuracy_graph

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#   Initialize Convolution Neural Network
cnn = ResNet()
print(cnn)
print(cnn.parameters())

#   Load Training Data
batch_size = 128
training_set_loader = CIFAR10DataLoader().get_data_loader(data_set='training', batch_size=batch_size)
#   Define Loss function
loss_func = nn.CrossEntropyLoss()
#   Initialize Optimizer
optimizer = optim.SGD(cnn.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-3)

if torch.cuda.is_available():
    cnn = cnn.cuda()

#   Train network model
#   Note: only TRAINING set will be used to train the network
#   Save Network Model per 5 epochs
plt_x = []
plt_train_y = []
for epoch in range(200):
    print('Epoch = ', epoch + 1)
    running_loss = 0.0
    mini_batch = 0
    #   Increase learning rate and weight_decay correspondingly
    #   Stage 1: epoch 1 - 50: lr = 1e-2, weight_decay = 5e-3
    #   Stage 2: epoch 51 - 70: lr = 1e-3, weight_decay = 5e-3
    #   Stage 3: epoch 71 up: lr = 1e-3, weight_decay = 1e-2
    if epoch == 50:
        optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-3)
    if epoch == 70:
        optimizer = optim.SGD(cnn.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-2)
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
    #   Training Accuracy Evaluation
    plt_x.append(epoch + 1)
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
    #   Save Network Model per 5 epochs
    if epoch % 5 == 4:
        torch.save(cnn, './model/ResNet-18_batch-size={}_epo_{}'.format(batch_size, epoch + 1))
print('Finished Training\n\n')

#   Show accuracy result
print(plt_x)
print(plt_train_y)
draw_accuracy_graph(plt_x, plt_train_y)
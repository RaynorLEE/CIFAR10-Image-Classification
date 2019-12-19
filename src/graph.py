import matplotlib.pyplot as plt

plt_x = []
plt_train_y = []
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(plt_x, plt_train_y, label='Training Accuracy')
plt.legend()
plt.title('Training Accuracy per Epoch')
plt.show()
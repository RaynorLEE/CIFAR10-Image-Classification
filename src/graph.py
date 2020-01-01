import matplotlib.pyplot as plt


def draw_accuracy_graph(plt_x: list, plt_y: list):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(plt_x, plt_y, label='Training Accuracy')
    plt.legend()
    plt.title('Training Accuracy per Epoch')
    plt.show()

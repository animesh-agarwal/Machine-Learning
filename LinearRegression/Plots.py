import matplotlib.pyplot as plt


def scatter_plot(x, y, size=10, x_label='x', y_label='y', color='b'):
    plt.scatter(x, y, s=size, color=color)
    set_labels(x_label, y_label)


def plot(x, y, x_label='x', y_label='y', color='r'):
    plt.plot(x, y, color=color)
    set_labels(x_label, y_label)


def ploty(y, x_label='x', y_label='y'):
    plt.plot(y)
    set_labels(x_label, y_label)


def set_labels(x_label, y_label):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

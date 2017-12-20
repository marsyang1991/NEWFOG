import matplotlib.pyplot as plt


def draw(y_pred, y_true):
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.ylim(-0.5, 1.5)
    plt.plot(y_pred, 'r*')
    plt.subplot(2, 1, 2)
    plt.ylim(-0.5, 1.5)
    plt.plot(y_true, 'b^')
    plt.show()


if __name__=="__main__":
    draw([1,0,1,0], [1,1,1,0])
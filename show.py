import matplotlib.pyplot as plt
import os

x_dir = 'img/test/'
y_dir = 'img/result/'


for x, y in zip(os.listdir(x_dir), os.listdir(y_dir)):
    plt.figure(dpi=300)
    plt.subplot(1, 2, 1)
    plt.imshow(plt.imread(x_dir+x))
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(plt.imread(y_dir+y))
    plt.axis('off')
    plt.savefig(x, bbox_inches='tight')

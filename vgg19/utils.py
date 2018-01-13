import numpy as np
import matplotlib.pyplot as plt   

def plot_im(im):
    plt.imshow(im,cmap='binary')
    plt.show()

def save_im(im, file):
    print('Saving image %s...'%(file))
    plt.imshow(im,cmap='binary')
    plt.savefig(file)


def load_image(imgpath):
    print('Loading image %s...'%(imgpath))
    from PIL import Image
    im = Image.open(imgpath)
    im = im.resize((224,224))
    im = np.array(im);
    return im

def load_label(file):
    label = list()
    with open(file, 'r') as f:
        for line in f:
            label.append(line)
    return label


'''
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.layers import Input
from keras.applications import VGG19
'''

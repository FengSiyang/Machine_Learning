import numpy as np
import matplotlib.pyplot as plt


def read_mnist():

    '''
    =========================== image data
    for image data, each byte (8 bits) stores one pixel, which is the data type, unit8.
    '''
    loaded_train = np.fromfile('mnist/train-images-idx3-ubyte', dtype='uint8')  # unsigned int 8-bits
    loaded_test = np.fromfile('mnist/t10k-images-idx3-ubyte', dtype='uint8')

    '''
    The first 16 bytes records the image dataset information:
    (0000 - 0015)
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    '''
    train_x = loaded_train[16:].reshape(60000, 28, 28)
    test_x = loaded_test[16:].reshape(10000, 28, 28)

    #print(train_x[0])

    '''
    =========================== label data
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    '''
    train_label = np.fromfile('mnist/train-labels-idx1-ubyte', dtype='uint8')
    test_label = np.fromfile('mnist/t10k-labels-idx1-ubyte', dtype='uint8')

    train_y = train_label[8:]
    test_y = test_label[8:]

    return train_x, train_y, test_x, test_y 




def plot_number(images, ind):
    '''
    plot one number in the images dataset
    '''
    plt.imshow(images[ind], cmap='gray')   # cmap: color map
    plt.axis('off')
    plt.show()


def plot_numbers(images, row, col):
    '''
    plot multiple numbers together
    '''
    show_numbers = np.vstack(np.split(np.hstack(images[:col*row]), row, axis=1))
    plt.imshow(show_numbers, cmap='gray')
    plt.axis('off')
    plt.show()





if __name__ == "__main__":
    train_x, train_y, test_x, test_y = read_mnist()
    #plot_number(10)
    plot_numbers(train_x, 4, 5)










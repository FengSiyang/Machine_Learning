"""
Keras:
based on Theano, TensorFlow and CDTK.
"""

from Read_MNIST import read_mnist, plot_number, plot_numbers

train_x, train_y, test_x, test_y = read_mnist()

# change pixel represent from [0-255] to [0, 1]:
train_x = train_x.reshape(-1, 28 * 28).astype(float) / 255  # change [28, 28] to [784], each sample for training
test_x = test_x.reshape(-1, 28 * 28).astype(float) / 255



import keras
from keras import Sequential, Model
from keras.layers import *



# ---------- Sequential Model
def sequential_model():
    model = Sequential([
        InputLayer(input_shape=(784,)),   # input layer
        Dense(10, activation='softmax')   # divided into 10 categories, and the probability of each category
    ])

    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    print(model.summary())

    hist = model.fit(train_x, train_y,
                        shuffle=True,     # the sequence of sample is not stable
                        batch_size=100,   # how many samples are used for training at one time
                        epochs=5,         # How many times to train all the samples
                        validation_data=(test_x, test_y))  # the validation samples, it's not good to use test data.
    '''
    # save model:
    model.save('lr.h5')
    # load model:
    from keras.models import load_model
    model = load_model('lr.h5')
    '''
    return hist



# -------------- Multilayer Perceptron
def multilayer_perceptron():
    model = Sequential([
        InputLayer(input_shape=(784,)),  # Input layer
        Dense(512, activation='relu'),   # insert a network, Dense(全连接) 512
        Dense(512, activation='relu'),   # insert .. active function: relu
        Dense(10, activation='softmax')  
    ])

    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
    print(model.summary())
    hist = model.fit(train_x, train_y,
                        shuffle=True,     # the sequence of sample is not stable
                        batch_size=100,   # how many samples are used for training at one time
                        epochs=5,         # How many times to train all the samples
                        validation_data=(test_x, test_y))  # the validation samples, it's not good to use test data.
    return hist



# --------------- Multilayer Perceptron, add Dropout
def MLP_dropout():
    model = Sequential([
        InputLayer(input_shape=(784,)),  # Input layer
        Dense(512, activation='relu'),   # insert a network, Dense(全连接) 512
        Dropout(0.2),                    # prevent overfitting
        Dense(512, activation='relu'),   # insert .. active function: relu
        BatchNormalization(),
        Dense(10, activation='softmax')  
    ])

    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')

    print(model.summary())
    hist = model.fit(train_x, train_y,
                        shuffle=True,     # the sequence of sample is not stable
                        batch_size=100,   # how many samples are used for training at one time
                        epochs=5,         # How many times to train all the samples
                        validation_data=(test_x, test_y))  # the validation samples, it's not good to use test data.
    return hist



# ----------------- CNN
def cnn():
    model = Sequential([
        InputLayer(input_shape=(784,)),
        Reshape(target_shape=(28, 28, 1)),   # size: 28*28, number of channels: 1
        Conv2D(filters=32, kernel_size=3, activation='relu'),
        Conv2D(filters=16, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(10, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')

    print(model.summary())
    hist = model.fit(train_x, train_y,
                        shuffle=True,     # the sequence of sample is not stable
                        batch_size=100,   # how many samples are used for training at one time
                        epochs=5,         # How many times to train all the samples
                        validation_data=(test_x, test_y))  # the validation samples, it's not good to use test data.
    return hist





# ========================================================================= #
# +++++++++++++++++ Other Model, not classification +++++++++++++++++++++++ #
# ========================================================================= #


# ---------------------- Autoencoder
'''
It can be used to fit the images into different size or other scenario.
x -> f(x) -> g(f(x)) ~ x
encoder: f()
decoder: g()
'''

def autoencoder():
    encoder = Sequential()
    encoder.add(Dense(512, input_dim=784, activation='relu'))   # input_dim is same to Inputlayer()
    encoder.add(Dense(256, activation='relu'))
    encoder.add(Dense(128, activation='relu'))

    decoder = Sequential()
    decoder.add(Dense(256, input_dim=128, activation='relu'))
    decoder.add(Dense(512, activation='relu'))
    decoder.add(Dense(784, activation='sigmoid'))   # sigmoid: map the real values into [0-1]

    model = Sequential()
    model.add(encoder)
    model.add(decoder)

    model.compile(loss='mse', optimizer='adam')
    model.summary()

    model.fit(train_x, train_x, 
                validation_split=0.2,
                shuffle=True,
                epochs=5)

    return model







def classification():
    #sequential_model()   # accuracy: 0.9237
    #multilayer_perceptron()  # accuracy: 0.9910
    #MLP_dropout()      # 0.9829
    cnn()      # 0.9904



def other_models():
    autoencoder()






if __name__ == "__main__":
    #classification()
    other_models()
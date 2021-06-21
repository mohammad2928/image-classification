from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.optimizers import Adam

class Models:

    def __init__(self):
        pass

    def LeNet(self, inputShape, numClasses, activation="relu", optimizer=Adam(learning_rate=5e-4)):
        # INPUT => CONV => RELU => POOL => CONV => RELU => POOL => FC => RELU => FC
        model = Sequential()
        # first conv
        model.add(Conv2D(5, 5, padding="same", input_shape=inputShape))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        #firast conv 
        model.add(Conv2D(5, 5, padding="same"))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # define the first FC => ACTIVATION layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation(activation))
        # define the second FC layer
        model.add(Dense(numClasses))
        # lastly, define the soft-max classifier
        model.add(Activation("softmax"))
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
        model.build()
        model.summary()
        return model


    def MLP(self, inputShape, numClasses, optimizer=Adam(learning_rate=5e-4)):
        model = Sequential()
        # two fully conected layer
        model.add(Dense(50, activation='relu', input_shape=inputShape))
        model.add(Dense(50, activation='relu', input_shape=inputShape))
        model.add(Flatten())
        # softmax layer
        model.add(Dense(numClasses, activation='softmax'))
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
        model.build()
        model.summary()
        return model

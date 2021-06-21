import keras
import tensorflow as tf
from keras.optimizers import Adam

#fit parameters
epochs = 1
batch_size = 10
verbose = 1
validation_split = 0.2


# test size for splitting test and train
test_split_size = 0.2

# list of optimizer for using in rps dataset

optimizer_list = {
    "sgd_momentum": keras.optimizers.SGD(momentum=0.01),
    "sgd": keras.optimizers.SGD(), 
    "adagrad": tf.keras.optimizers.Adagrad(),
    "rmsprob": keras.optimizers.RMSprop(learning_rate=0.00001, decay=1e-6),
    "adam": Adam(learning_rate=5e-4)
}
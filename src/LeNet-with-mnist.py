import tensorflow as tf
from keras.utils.np_utils import to_categorical
from models import Models
from utils import plot
from parameters import batch_size, epochs, batch_size, validation_split, verbose


def main():
    # load data 
    # in the first time, it will be downloaded. 
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # print dimentions
    print("shape of train-set is: ", train_images.shape)
    print("shape of test-set is: ", test_images.shape)
    print("The number of classes is ", len(set(test_labels)))

    # change lables to catagory 
    y_train = []
    for i in train_labels:
        y_train.append(int(i))
    y_train = to_categorical(y_train)

    y_test = []
    for i in test_labels:
        y_test.append(int(i))
    y_test = to_categorical(y_test)

    # reshape images to 28*28*1 
    # convert to 3-D
    X_train = train_images.reshape(
        train_images.shape[0],
        train_images.shape[1],
        train_images.shape[2],
         1
    )
    X_test = test_images.reshape(
        test_images.shape[0],
        test_images.shape[1],
        test_images.shape[2],
        1
    )

    # parameters
    input_shape = X_train.shape[1:]
    num_class = len(set(test_labels))

    # initiate the models
    Model = Models() 
    model = Model.LeNet(input_shape, num_class)

    # fit model
    histoey = model.fit(
    X_train, y_train,
    epochs=epochs,
    verbose=verbose,
    batch_size=batch_size,
    validation_split= validation_split,
    )

    # plot training phase
    plot(histoey, "LeNet")

    # print accuracy and loss
    out = model.evaluate(X_test, y_test)


if __name__ == '__main__':
    main()
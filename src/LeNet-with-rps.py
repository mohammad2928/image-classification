import tensorflow as tf
from keras.utils.np_utils import to_categorical
from models import Models
from utils import plot
from parameters import epochs, validation_split, verbose, test_split_size, optimizer_list
import glob
from PIL import Image
import os
import numpy as np
import random 


def main():
    # download rps data ind extract it into a folder by the name data so the data will be as follow ./data/rps
    # read data
    classes = {
    "paper": 0,
    "rock": 1,
    "scissors": 2,
    }
    paper_images = []
    for image in glob.glob("data/rps/paper/*.png"):
        paper_images.append(
            np.array(
                Image.open(image)
            )
        )
        
    rock_images = []
    for image in glob.glob("data/rps/rock/*.png"):
        rock_images.append(
            np.array(
                Image.open(image)
            )
        )

    scissors_images = []
    for image in glob.glob("data/rps/scissors/*.png"):
        scissors_images.append(
            np.array(
                Image.open(image)
            )
        )

    # shuffle data
    all_data = []
    for image in paper_images:
        all_data.append((image, 0))

    for image in rock_images:
        all_data.append((image, 1))
        
    for image in scissors_images:
        all_data.append((image, 2))

    random.shuffle(all_data)

    # split test and train

    X_test = []
    y_test = []
    for item in all_data[:int(len(all_data)*test_split_size)]:
        X_test.append(item[0])
        y_test.append(item[1])

    X_train = []
    y_train = []
    for item in all_data[int(len(all_data)*test_split_size):]:
        X_train.append(item[0])
        y_train.append(item[1])

    X_test = np.array(X_test)
    X_train = np.array(X_train)

    # print dimentions
    print("shape of train-set is: ", X_train.shape)
    print("shape of test-set is: ", X_test.shape)
    
    # change lables to catagory 
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # parameters
    input_shape = X_train.shape[1:]
    num_class = 3

    # initiate the models
    Model = Models() 
    model = Model.LeNet(input_shape, num_class)

    # fit model
    for k,v in optimizer_list.items():
        model = Model.LeNet(input_shape, num_class, optimizer=v)
        histoey = model.fit(
                X_train, y_train,
                epochs=epochs,
                verbose=verbose,
                validation_split= validation_split,
            )
        plot(histoey, k)
        out = model.evaluate(X_test, y_test)
        print("for ", k, " the accuracy and loss is ", out)

if __name__ == '__main__':
    main()
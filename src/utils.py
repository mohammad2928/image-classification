
import matplotlib.pyplot as plt


def plot(history, out_name):
    # history of trained model
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig("{}-acc.png".format(out_name))
    # summarize history for loss
    plt.show() 
    
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.savefig("{}-loss.png".format(out_name))
    plt.show()



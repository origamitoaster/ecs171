#Problem 2
import keras
from keras import initializers
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def encodeData(class_labels):
    #Performs One Hot Encoding on class labels.
    vals = array(class_labels)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(vals)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False, n_values=10)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return(onehot_encoded)


def createModel():
    # define the architecture of the network
    model = Sequential()
    model.add(Dense(3, input_dim=8, activation='sigmoid', bias_initializer=initializers.RandomNormal(seed=10))) #8 inputs, what about the 9th for bias?c
    model.add(Dense(3, activation='sigmoid', bias_initializer=initializers.RandomNormal(seed=10)))
    model.add(Dense(10, activation='softmax', bias_initializer=initializers.RandomNormal(seed=10)))
    #model.add(Activation("softmax"))
    sgd = SGD(lr=0.1)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    return model

def train_valModel(model, train_data, train_label, test_data, test_label, epoch):
    history = LossHistory()
    model.fit(train_data, train_label, epochs=epoch, batch_size=1, verbose=1, callbacks=[history], validation_data=(test_data, test_label))
    return history

def trainModel(model, train_data, train_label, epoch):
    model.fit(train_data, train_label, epochs=epoch, batch_size=1, verbose=1)
    return model

def testModel(model, test_data, test_label):
    print("[INFO] evaluating on testing set...")
    (loss, accuracy) = model.evaluate(test_data, test_label, batch_size=1, verbose=1)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
    return accuracy

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.error = []
        self.val_error = []
        self.i_weights = []
        self.e_weights =[]
        
    def on_epoch_end(self, batch, logs={}):
        self.error.append(1 - logs.get('acc'))
        self.val_error.append(1 - logs.get('val_acc'))
        e_weights = [self.model.layers[2].get_weights()[0][0][0], self.model.layers[2].get_weights()[0][1][0],
                   self.model.layers[2].get_weights()[0][2][0], self.model.layers[2].get_weights()[1][0]]
        self.e_weights.append(e_weights)
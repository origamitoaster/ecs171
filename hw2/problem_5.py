#Problem 5
from keras import initializers
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense

def create_1hModel(hidden_node):
    # define the architecture of the network
    model = Sequential()
    model.add(Dense(hidden_node, input_dim=8, activation='sigmoid', bias_initializer=initializers.RandomNormal(seed=10))) #8 inputs, what about the 9th for bias?c
    model.add(Dense(10, activation='softmax', bias_initializer=initializers.RandomNormal(seed=10)))
    #model.add(Activation("softmax"))
    sgd = SGD(lr=0.1)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    return model

def create_2hModel(hidden_node):
    # define the architecture of the network
    model = Sequential()
    model.add(Dense(hidden_node, input_dim=8, activation='sigmoid', bias_initializer=initializers.RandomNormal(seed=10))) #8 inputs, what about the 9th for bias?c
    model.add(Dense(hidden_node, activation='sigmoid', bias_initializer=initializers.RandomNormal(seed=10)))
    model.add(Dense(10, activation='softmax', bias_initializer=initializers.RandomNormal(seed=10)))
    #model.add(Activation("softmax"))
    sgd = SGD(lr=0.1)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    return model

def create_3hModel(hidden_node):
    # define the architecture of the network
    model = Sequential()
    model.add(Dense(hidden_node, input_dim=8, activation='sigmoid', bias_initializer=initializers.RandomNormal(seed=10))) #8 inputs, what about the 9th for bias?c
    model.add(Dense(hidden_node, activation='sigmoid', bias_initializer=initializers.RandomNormal(seed=10)))
    model.add(Dense(hidden_node, activation='sigmoid', bias_initializer=initializers.RandomNormal(seed=10)))
    model.add(Dense(10, activation='softmax', bias_initializer=initializers.RandomNormal(seed=10)))
    #model.add(Activation("softmax"))
    sgd = SGD(lr=0.1)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    return model
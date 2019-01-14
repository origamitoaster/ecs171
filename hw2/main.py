#add current folder to path to use functions from other files
import os
import sys
#   csfp -> current script folder path
csfp = os.path.abspath(os.path.dirname(__file__))
if csfp not in sys.path:
    sys.path.insert(0, csfp)
#import it and invoke it by one of the ways described above

#import my functions
from problem_1 import LOF
from problem_1 import iso_forest
from problem_2 import createModel
from problem_2 import trainModel
from problem_2 import testModel
from problem_2 import train_valModel
from problem_2 import encodeData
from problem_5 import create_1hModel
from problem_5 import create_2hModel
from problem_5 import create_3hModel

#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Import data
yeast = pd.read_csv('yeast.data', delim_whitespace=True)
yeast.columns = ["seq_name", "mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc", "class"]

#Problem 1
yeast_lof = LOF(yeast)
yeast_if = iso_forest(yeast)

def p1():
    yeast_lof.iloc[:,1].plot() #blue
    yeast_if.iloc[:,1].plot() #orange
    plt.show()

#Problem 2
#Randomly shuffle and split dataset into training and testing sets
from sklearn.model_selection import train_test_split
y_train, y_test = train_test_split(yeast_lof, test_size=.3, train_size=.7, random_state=10, shuffle=True)
#Perform onehot encoding on testing and training labels
train_label = encodeData(y_train['class'])
test_label = encodeData(y_test['class'])
def p2():
    #Create ANN using keras 
    model = createModel()
    history = train_valModel(model, y_train.iloc[:,1:9], train_label, y_test.iloc[:,1:9], test_label, 100)
    #testModel(trained_model, y_test.iloc[:,1:9], test_label)

    #Plot CYT related graphs
    plt.figure(figsize=(20,10))
    plt.plot(history.error)
    plt.plot(history.val_error)
    plt.title('CYT Training and Validation Error')
    plt.ylabel('Error')
    plt.xlabel('Epoch')
    plt.legend(('error', 'val_error'), loc='upper left')
    plt.savefig("cyt_error.png")
    plt.show()

    plt.figure(figsize=(20,10))
    plt.plot(history.e_weights)
    plt.title('CYT Weights and Bias')
    plt.ylabel('Weight')
    plt.xlabel('Epoch')
    plt.legend(('h1', 'h2', 'h3', 'bias'), loc='upper left')
    plt.savefig("cyt_weights.png")
    plt.show()

#Problem 3
def p3():
    yeast_label = encodeData(yeast['class'])
    full_model = createModel()
    full_model = trainModel(full_model, yeast.iloc[:,1:9], yeast_label, 100)
    weights = full_model.get_weights()
    print('Weights:')
    print(weights)
    return full_model

#Problem 4
def p4():
    model.layers[0].set_weights([array([
        [0 , 0, 0],
        [0 , 0, 0],
        [0 , 0, 0],
        [0 , 0, 0],
        [0 , 0, 0],
        [0 , 0, 0],
        [0 , 0, 0],
        [0 , 0, 0]]),
    array([1 , 1, 1])])
    model.layers[1].set_weights([array([
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0]]),
    array([1, 1, 1])])
    model.layers[2].set_weights([array([
        [1, 0, 0, 0,  0, 0,  0,  0,  0, 0],
        [1, 0, 0, 0,  0, 0,  0,  0,  0, 0],
        [1, 0, 0, 0,  0, 0,  0,  0,  0, 0]]),
    array([1, 1, 1, 1,  1, 1,  1,  1,  1, 1])])

#Problem 5
#   Nodes
#   3   6   9   12
#                   1
#                   2   Layers
#                   3
def p5():
    grid = np.zeros((3,4))
    for i in range(3, 15, 3):
        idx = (i//3 - 1)
        hidden_1 = trainModel(create_1hModel(i), y_train.iloc[:,1:9], train_label, 100)
        grid[0, idx] = 1 - testModel(hidden_1, y_test.iloc[:,1:9], test_label)

        hidden_2 = trainModel(create_2hModel(i), y_train.iloc[:,1:9], train_label, 100)
        grid[1, idx] = 1 - testModel(hidden_2, y_test.iloc[:,1:9], test_label)

        hidden_3 = trainModel(create_3hModel(i), y_train.iloc[:,1:9], train_label, 100)
        grid[2, idx] = 1 - testModel(hidden_3, y_test.iloc[:,1:9], test_label)
    print('Grid Search:')
    print(grid)

def p6(model):
    unknown = np.array([0.52, 0.47, 0.52, 0.23, 0.55, 0.03, 0.52, 0.39])
    unknown.shape = (1,8)
    labels = ['CYT', 'EXC', 'ME1', 'ME2', 'ME3', 'MIT', 'NUC', 'POX', 'VAC']
    print(labels[np.argmax(model.predict(unknown, verbose=1))])

p1()
p2()
model = p3()
p5()
p6(model)


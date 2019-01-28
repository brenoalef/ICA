import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from single_layer_perceptron import SingleLayerPerceptron
import time


iris_datasets = datasets.load_iris()
X = iris_datasets.data
Y = np.zeros((iris_datasets.target.shape[0], 3))
for i in range(iris_datasets.target.shape[0]):
    Y[i, iris_datasets.target[i]] = 1

accuracy = np.zeros((20, 1))
mean_time = 0
for i in range(20):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.80)
    
    start_time = time.clock()
    perceptron = SingleLayerPerceptron(num_features = 4)
    perceptron.fit(X_train, Y_train)
    Y_hat = perceptron.predict(X_test)
    mean_time += (time.clock() - start_time)/20    

    accuracy[i] = 1 - np.sum(np.abs(np.sum(Y_hat - Y_test))) / Y_hat.size

print("Mean execution time", mean_time)
print("Accuracy", np.mean(accuracy))
print("Standard Deviation (accuracy)", np.std(accuracy, axis=0))

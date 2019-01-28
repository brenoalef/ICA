import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from single_layer_perceptron import SingleLayerPerceptron

dataset = np.array([[np.random.uniform(4, 6), x1, 1, 0, 0] for x1 in np.random.uniform(14, 15, 50)])
dataset = np.append(dataset, [[x1, np.random.uniform(4, 6), 0, 1, 0] for x1 in np.random.uniform(9, 11, 50)], axis=0)
dataset = np.append(dataset, [[np.random.uniform(14, 15), x1, 0, 0, 1] for x1 in np.random.uniform(14, 15, 50)], axis=0)

accuracy = np.zeros((20, 1))
for i in range(20):
    X_train, X_test, Y_train, Y_test = train_test_split(dataset[:, :2],
    dataset[:, 2:5], test_size=0.50)
    perceptron = SingleLayerPerceptron(num_features = 2)
    perceptron.fit(X_train, Y_train)
    Y_hat = perceptron.predict(X_test)
    accuracy[i] = 1 - np.sum(np.abs(np.sum(Y_hat - Y_test))) / Y_hat.size

print("Accuracy", np.mean(accuracy))
print("Standard Deviation (accuracy)", np.std(accuracy, axis=0))

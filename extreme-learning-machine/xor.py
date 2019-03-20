import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from elm import ELM
import time
from mlxtend.plotting import plot_decision_regions


dataset = np.array([[x1, np.random.uniform(-0.2, 0.2), 0] for x1 in np.random.uniform(-0.2, 0.2, 50)])
dataset = np.append(dataset, [[x1, np.random.uniform(-0.2, 0.2), 1] for x1 in np.random.uniform(0.8, 1.2, 50)], axis=0)
dataset = np.append(dataset, [[np.random.uniform(-0.2, 0.2), x1, 1] for x1 in np.random.uniform(0.8, 1.2, 50)], axis=0)
dataset = np.append(dataset, [[x1, np.random.uniform(0.8, 1.2), 0] for x1 in np.random.uniform(0.8, 1.2, 50)], axis=0)

iters = 20
accuracy = np.zeros((iters, 1))
mean_time = 0
for i in range(iters):
    X_train, X_test, Y_train, Y_test = train_test_split(dataset[:, :2],
    dataset[:, 2], test_size=0.10)
    Y_train = Y_train.reshape((-1, 1))
    Y_test = Y_test.reshape((-1, 1))

    start_time = time.clock()
    elm = ELM(hidden_units = 8, activation="log")
    #elm = ELM(hidden_units = 8, activation="tan")
    #elm = ELM(hidden_units = 8, activation="relu")
    elm.fit(X_train, Y_train)
    Y_hat = elm.predict(X_test)
    Y_hat = np.round(Y_hat)
    mean_time += (time.clock() - start_time)/iters    

    accuracy[i] = np.sum(np.where(Y_hat == Y_test, 1, 0))/len(Y_test)

print("Mean execution time", mean_time)
print("Accuracy", np.mean(accuracy))
print("Standard Deviation (accuracy)", np.std(accuracy, axis=0))

plot_decision_regions(dataset[:, :2], dataset[:, -1].astype(np.integer), clf=elm, legend=2)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("XOR")
plt.show()

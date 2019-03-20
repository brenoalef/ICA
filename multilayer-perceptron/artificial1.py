import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from mlp import MLP
from mlxtend.plotting import plot_decision_regions


dataset = np.array([[x1, np.random.uniform(-0.2, 0.2), 0] for x1 in np.random.uniform(-0.2, 0.2, 50)])
dataset = np.append(dataset, [[x1, np.random.uniform(-0.2, 0.2), 1] for x1 in np.random.uniform(0.8, 1.2, 50)], axis=0)
dataset = np.append(dataset, [[np.random.uniform(-0.2, 0.2), x1, 1] for x1 in np.random.uniform(0.8, 1.2, 50)], axis=0)
dataset = np.append(dataset, [[x1, np.random.uniform(0.8, 1.2), 0] for x1 in np.random.uniform(0.8, 1.2, 50)], axis=0)

iters = 1
accuracy = np.zeros((iters, 1))
for i in range(iters):
    X_train, X_test, Y_train, Y_test = train_test_split(dataset[:, :2],
    dataset[:, 2], test_size=0.10)
    Y_train = Y_train.reshape((-1, 1))
    Y_test = Y_test.reshape((-1, 1))

    mlp = MLP(hidden_units = 10, activation="log", lr=1.0e-6, n_iter=100)
    #mlp = MLP(hidden_units = 8, activation="tan", lr=0.01, n_iter=20)
    mlp.fit(X_train, Y_train)
    Y_hat = mlp.predict(X_test)
    print(Y_hat)
    Y_hat = np.round(Y_hat) 

    accuracy[i] = np.sum(np.where(Y_hat == Y_test, 1, 0))/len(Y_test)

print("Accuracy", np.mean(accuracy))
print("Standard Deviation (accuracy)", np.std(accuracy, axis=0))

plot_decision_regions(dataset[:, :2], dataset[:, -1].astype(np.integer), clf=mlp, legend=2)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Artificial 1")
plt.show()

'''
X = dataset[:, :2]
Y = dataset[:, 2]

iters = 1
accuracy = []
n_folds = 10
lr_values = [0.01, 0.001, 0.0001, 0.00001]
for i in range(iters):
    CVO = KFold(n_splits=n_folds)
    acc = []
    for train_index, test_index in CVO.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        Y_train = Y_train.reshape((-1, 1))
        Y_test = Y_test.reshape((-1, 1))

        mlp = MLP(hidden_units = 2, activation="log", lr=lr_values[i], n_iter=100)
        mlp.fit(X_train, Y_train)
        Y_hat = mlp.predict(X_test)
        Y_hat = np.round(Y_hat)
        acc.append(np.sum(np.where(Y_hat == Y_test, 1, 0))/len(Y_test))   
    accuracy.append(acc)

print("Learning ratings:", lr_values)
print("Accuracy:", np.mean(accuracy, axis=0))
print("Standard Deviation (accuracy):", np.std(accuracy, axis=0))

plot_decision_regions(dataset[:, :2], dataset[:, -1].astype(np.integer), clf=mlp, legend=2)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Artificial 1")
plt.show()
'''

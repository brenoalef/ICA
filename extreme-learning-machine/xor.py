import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from elm import ELM
import time
from mlxtend.plotting import plot_decision_regions,plot_confusion_matrix


dataset = np.array([[x1, np.random.uniform(-0.2, 0.2), 0] for x1 in np.random.uniform(-0.2, 0.2, 50)])
dataset = np.append(dataset, [[x1, np.random.uniform(-0.2, 0.2), 1] for x1 in np.random.uniform(0.8, 1.2, 50)], axis=0)
dataset = np.append(dataset, [[np.random.uniform(-0.2, 0.2), x1, 1] for x1 in np.random.uniform(0.8, 1.2, 50)], axis=0)
dataset = np.append(dataset, [[x1, np.random.uniform(0.8, 1.2), 0] for x1 in np.random.uniform(0.8, 1.2, 50)], axis=0)

X = dataset[:, :2]
Y = dataset[:, 2:3]

iters = 20
n_folds = 3
accuracy = np.zeros((20, 1))
mean_time = 0
data = []
best = [[], 0, None]
for i in range(iters):
    CVO = KFold(n_splits=n_folds, shuffle=True)
    acc_values = []
    for train_index, test_index in CVO.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        elm = ELM(hidden_units = 8, activation="log")
        #elm = ELM(hidden_units = 8, activation="tan")
        elm.fit(X_train, Y_train)
        Y_hat = elm.predict(X_test)
        Y_hat = np.round(Y_hat)  
        acc_values.append(np.sum(np.where(Y_hat == Y_test, 1, 0))/len(Y_test))
        if acc_values[-1] > best[1]:
            best[0] = confusion_matrix(Y_test, Y_hat)
            best[1] = acc_values[-1]
            best[2] = elm
    accuracy[i] = np.mean(acc_values)

print("Accuracy", np.mean(accuracy))
print("Standard Deviation (accuracy)", np.std(accuracy, axis=0))

conf_matrix = best[0]
print("Confusion Matrix")
print(conf_matrix)

fig, ax = plot_confusion_matrix(conf_mat=conf_matrix)
plt.xlabel("Predicted")
plt.ylabel("Desired")
plt.show()

plot_decision_regions(dataset[:, :2], dataset[:, -1].astype(np.integer), clf=best[2], legend=2)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("XOR")
plt.show()

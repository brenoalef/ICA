import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from mlp import MLP
import time
from mlxtend.plotting import plot_decision_regions


iris_datasets = datasets.load_iris()
X = iris_datasets.data
Y = np.zeros((iris_datasets.target.shape[0], 3))
for i in range(iris_datasets.target.shape[0]):
    Y[i, iris_datasets.target[i]] = 1

accuracy = np.zeros((20, 1))
mean_time = 0
data = []
for i in range(20):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.80)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    mlp = MLP(hidden_units = 10, activation="log", lr=0.01, n_iter=20)
    #mlp = MLP(hidden_units = 8, activation="tan", lr=0.01, n_iter=20)
    mlp.fit(X_train, Y_train)
    Y_hat = mlp.predict(X_test)  

    accuracy[i] = 1 - np.sum(np.abs(np.sum(Y_hat - Y_test))) / Y_hat.size
    data.append([X_train, X_test, Y_train, Y_test, Y_hat])

print("Accuracy", np.mean(accuracy))
print("Standard Deviation (accuracy)", np.std(accuracy, axis=0))
X_train, X_test, Y_train, Y_test, Y_hat = data[(np.abs(accuracy - np.mean(accuracy))).argmin()]


conf_matrix = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(Y_hat, axis=1))
print("Confusion Matrix", conf_matrix)

labels = ["Others", "Setosa"]
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_matrix)
plt.title("Confusion Matrix")
fig.colorbar(cax)
ax.set_xticklabels([""] + labels)
ax.set_yticklabels([""] + labels)
plt.xlabel("Predicted")
plt.ylabel("Desired")
plt.show()

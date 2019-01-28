import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from perceptron import Perceptron


iris_datasets = datasets.load_iris()
X = iris_datasets.data
Y = np.zeros((iris_datasets.target.shape[0], 1))
Y[iris_datasets.target != 0] = 0
Y[iris_datasets.target == 0] = 1

accuracy = np.zeros((20, 1))
data = []
for i in range(20):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.80)
    Y_train = Y_train.reshape((X_train.shape[0], 1))
    Y_test = Y_test.reshape((X_test.shape[0], 1))
    perceptron = Perceptron(num_features = 4)
    perceptron.fit(X_train,Y_train)
    Y_hat = perceptron.predict(X_test)

    accuracy[i] = 1 - np.sum(np.abs(Y_hat - Y_test)) / Y_hat.size
    data.append([X_train, X_test, Y_train, Y_test])

print("Accuracy", np.mean(accuracy))
X_train, X_test, Y_train, Y_test = data[(np.abs(accuracy - np.mean(accuracy))).argmin()]


cm_bright = ListedColormap(["#0000FF", "#FF0000"])
plt.figure(figsize=(7,5))
plt.scatter(X_train[:,0], X_train[:,1], c=Y_train[:, 0], cmap=cm_bright)
plt.scatter(None, None, color = "r", label="Setosa")
plt.scatter(None, None, color = "b", label="Others")
plt.legend()
plt.title("Train data")
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.show()

cm_bright = ListedColormap(["#0000FF", "#FF0000"])
plt.figure(figsize=(7,5))
plt.scatter(X_test[:,0], X_test[:,1], c=Y_test[:, 0], cmap=cm_bright)
plt.scatter(None, None, color = "r", label="Setosa")
plt.scatter(None, None, color = "b", label="Others")
plt.legend()
plt.title("Test data")
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.show()

conf_matrix = confusion_matrix(Y_test, Y_hat)
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
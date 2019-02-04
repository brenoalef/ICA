import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from perceptron import Perceptron
import time


dataset = np.array([[x1, np.random.uniform(-1, 1), 0] for x1 in np.random.uniform(-1, 1, 10)])
dataset = np.append(dataset, [[x1, np.random.uniform(-1, 1), 0] for x1 in np.random.uniform(14, 16, 10)], axis=0)
dataset = np.append(dataset, [[np.random.uniform(-1, 1), x1, 0] for x1 in np.random.uniform(14, 16, 10)], axis=0)
dataset = np.append(dataset, [[x1, np.random.uniform(14, 15), 1] for x1 in np.random.uniform(10, 15, 10)], axis=0)

accuracy = np.zeros((20, 1))
data = []
mean_time = 0
for i in range(20):
    X_train, X_test, Y_train, Y_test = train_test_split(dataset[:, :2],
    dataset[:, 2], test_size=0.10)
    Y_train = Y_train.reshape((X_train.shape[0], 1))
    Y_test = Y_test.reshape((X_test.shape[0], 1))

    start_time = time.clock()
    perceptron = Perceptron()
    perceptron.fit(X_train,Y_train)
    Y_hat = perceptron.predict(X_test)
    mean_time += (time.clock() - start_time)/20    

    accuracy[i] = 1 - np.sum(np.abs(Y_hat - Y_test)) / Y_hat.size
    data.append([X_train, X_test, Y_train, Y_test, Y_hat])

print("Mean execution time", mean_time)
print("Accuracy", np.mean(accuracy))
X_train, X_test, Y_train, Y_test, Y_hat = data[(np.abs(accuracy - np.mean(accuracy))).argmin()]
'''
cm_bright = ListedColormap(["#0000FF", "#FF0000"])
plt.figure(figsize=(7,5))
plt.scatter(X_train[:,0], X_train[:,1], c=Y_train[:, 0], cmap=cm_bright)
plt.scatter(None, None, color = "b", label="Class 0")
plt.scatter(None, None, color = "r", label="Class 1")
plt.legend()
plt.title("Train data")
plt.show()

cm_bright = ListedColormap(["#0000FF", "#FF0000"])
plt.figure(figsize=(7,5))
plt.scatter(X_test[:,0], X_test[:,1], c=Y_test[:, 0], cmap=cm_bright)
plt.scatter(None, None, color = "b", label="Class 0")
plt.scatter(None, None, color = "r", label="Class 1")
plt.legend()
plt.title("Test data")
plt.show()
'''
conf_matrix = confusion_matrix(Y_test, Y_hat)
print("Confusion Matrix", conf_matrix)
'''
labels = ["Class 0", "Class 1"]
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
'''
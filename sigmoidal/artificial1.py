import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sigmoidal import Sigmoidal
import time


dataset = np.array([[np.random.uniform(4, 6), x1, 1, 0, 0] for x1 in np.random.uniform(14, 15, 50)])
dataset = np.append(dataset, [[np.random.uniform(14, 15), x1, 0, 1, 0] for x1 in np.random.uniform(14, 15, 50)], axis=0)
dataset = np.append(dataset, [[x1, np.random.uniform(4, 6), 0, 0, 1] for x1 in np.random.uniform(9, 11, 50)], axis=0)

accuracy = np.zeros((20, 1))
mean_time = 0
for i in range(20):
    X_train, X_test, Y_train, Y_test = train_test_split(dataset[:, :2], dataset[:, 2:5], test_size=0.50)

    start_time = time.clock()
    #sig = Sigmoidal(num_features = 2, type="log")
    sig = Sigmoidal(num_features = 2)
    sig.fit(X_train, Y_train)
    Y_hat = sig.predict(X_test)
    mean_time += (time.clock() - start_time)/20    

    accuracy[i] = 1 - np.sum(np.abs(np.sum(Y_hat - Y_test))) / Y_hat.size

print("Mean execution time", mean_time)
print("Accuracy", np.mean(accuracy))
print("Standard Deviation (accuracy)", np.std(accuracy, axis=0))

x_min, x_max = dataset[:, 0].min() - 2., dataset[:, 0].max() + .5
y_min, y_max = dataset[:, 1].min() - 2., dataset[:, 1].max() + .5

xx = np.array([[x, y] for x in np.arange(x_min, x_max, 0.1) for y in np.arange(y_min, y_max, 0.1)])
Z = sig.predict(xx)

cm_dark = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
plt.figure(figsize=(7,5))
plt.scatter(xx[:, 0], xx[:, 1], c=Z.argmax(axis=1), cmap=cm_dark)

classes = dataset[:, 2:5].argmax(axis=1)
plt.scatter(dataset[classes == 0, 0], dataset[classes == 0, 1], s=30,  marker="o", c="#550000")
plt.scatter(dataset[classes == 1, 0], dataset[classes == 1, 1], s=30, marker="^", c="#005500")
plt.scatter(dataset[classes == 2, 0], dataset[classes == 2, 1], s=30, marker="*", c="#000055")

plt.scatter(None, None, color = "r", label="Class [1, 0, 0]")
plt.scatter(None, None, color = "g", label="Class [0, 1, 0]")
plt.scatter(None, None, color = "b", label="Class [0, 0, 1]")
leg = plt.legend(loc='lower right')

plt.title("Dataset")
plt.xlabel("X1")
plt.ylabel("x2")
plt.show()

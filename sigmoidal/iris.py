import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sigmoidal import Sigmoidal
import time


iris_datasets = datasets.load_iris()
X = iris_datasets.data
Y = np.zeros((iris_datasets.target.shape[0], 3))
for i in range(iris_datasets.target.shape[0]):
    Y[i, iris_datasets.target[i]] = 1

accuracy = np.zeros((20, 1))
mean_time = 0
for i in range(1):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.80)

    start_time = time.clock()
    sig = Sigmoidal(num_features = 4)
    #sig = Sigmoidal(num_features = 4, type="log")
    sig.fit(X_train, Y_train)
    Y_hat = sig.predict(X_test)
    mean_time += (time.clock() - start_time)/20    

    accuracy[i] = 1 - np.sum(np.abs(np.sum(Y_hat - Y_test))) / Y_hat.size

print("Mean execution time", mean_time)
print("Accuracy", np.mean(accuracy))
print("Standard Deviation (accuracy)", np.std(accuracy, axis=0))

n_classes = 3
plot_colors = ["#AA0000", "#00AA00", "#0000AA"]
plot_step = 0.1
cm_dark = ListedColormap(['#FFCCCC', '#CCFFCC', '#CCCCFF'])
cm_bright = ListedColormap(plot_colors)
for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):
    clf = Sigmoidal(num_features = 2)
    #clf = Sigmoidal(num_features = 2, type="log")
    clf.fit(X_train[:, pair], Y_train)
    
    plt.subplot(2, 3, pairidx + 1)
    
    x_min, x_max = X[:, pair[0]].min() - 1, X[:, pair[0]].max() + 1
    y_min, y_max = X[:, pair[1]].min() - 1, X[:, pair[1]].max() + 1
    xx = np.array([[x, y] for x in np.arange(x_min, x_max, plot_step) for y in np.arange(y_min, y_max, plot_step)])
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = clf.predict(xx)

    plt.scatter(xx[:, 0], xx[:, 1], c=Z.argmax(axis=1), cmap=cm_dark)

    plt.xlabel(iris_datasets.feature_names[pair[0]])
    plt.ylabel(iris_datasets.feature_names[pair[1]])

    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(Y.argmax(axis=1) == i)
        plt.scatter(X[idx, pair[0]], X[idx, pair[1]], c=color, label=iris_datasets.target_names[i],
                    cmap=cm_bright, edgecolor='black', s=15)
    


plt.suptitle("Decision surface using paired features")
plt.legend(loc='lower right', borderpad=0, handletextpad=0)
plt.axis("tight")
plt.show()


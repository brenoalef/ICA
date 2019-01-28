import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from adaline import Adaline
import time


def f(x1, x2):
    return 3*x1 - 2*x2 + 5 + np.random.rand()


dataset = np.array([[x1, x2, f(x1, x2)] for x1 in np.random.uniform(-30, 30, 20) for x2 in np.random.uniform(-30, 30, 20)])

mse = np.zeros((20, 1))
rmse = np.zeros((20, 1))
mean_time = 0
for i in range(20):
    X_train, X_test, Y_train, Y_test = train_test_split(dataset[:, :2], dataset[:, 2], test_size=0.80)
    Y_train = Y_train.reshape((Y_train.shape[0], 1))
    Y_test = Y_test.reshape((Y_test.shape[0], 1))

    start_time = time.clock()
    adaline = Adaline(eta = 0.00001, n_iter=400)
    adaline.fit(X_train,Y_train)
    Y_hat = adaline.predict(X_test)
    mean_time += (time.clock() - start_time)/20    

    mse[i] = ((Y_test - Y_hat)**2).mean(axis=0)
    rmse[i] = mse[i]**(1./2)

print("Mean execution time", mean_time)
print("Standard Deviation (MSE)", np.std(mse, axis=0))
print("Standard Deviation (RMSE)",np.std(rmse, axis=0))

cm_bright = ListedColormap(["#0000FF", "#FF0000"])
plt.figure(figsize=(7,5))
plt.scatter(X_test[:,0], X_test[:,1], c=[1 if y>= 0 else 0 for y in Y_test[:, 0]], cmap=cm_bright)
plt.scatter(None, None, color = "r", label="f(x1, x2) >= 0")
plt.scatter(None, None, color = "b", label="f(x1, x2) < 0")
plt.legend()
plt.title("Visualize the data")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
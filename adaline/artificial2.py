import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from adaline import Adaline
import time


def f(x1, x2):
    return 3*x1 - 2*x2 + 5 + np.random.uniform(-0.5, 0.5)


dataset = np.array([[x1, x2, f(x1, x2)] for x1 in np.random.uniform(-30, 30, 20) for x2 in np.random.uniform(-30, 30, 20)])
min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(dataset)
dataset = min_max_scaler.transform(dataset)

mse = np.zeros((20, 1))
rmse = np.zeros((20, 1))
mean_time = 0
#cost = []
for i in range(20):
    X_train, X_test, Y_train, Y_test = train_test_split(dataset[:, :2], dataset[:, 2], test_size=0.80)
    Y_train = Y_train.reshape((Y_train.shape[0], 1))
    Y_test = Y_test.reshape((Y_test.shape[0], 1))

    start_time = time.clock()
    adaline = Adaline(eta = 0.01, n_iter=200)
    adaline.fit(X_train,Y_train)
    Y_hat = adaline.predict(X_test)
    mean_time += (time.clock() - start_time)/20    

    mse[i] = ((Y_test - Y_hat)**2).mean(axis=0)
    rmse[i] = mse[i]**(1./2)
    #cost.append(adaline.error) 

print("Mean execution time", mean_time)
print("Standard Deviation (MSE)", np.std(mse, axis=0))
print("Standard Deviation (RMSE)",np.std(rmse, axis=0))

'''
fig, ax = plt.subplots()
plt.plot(range(1, len(cost[0]) + 1), cost[0], "o-")
plt.title("Cost")
plt.xlabel("epoch")
plt.ylabel("cost")
plt.show()
'''

xx = np.array([[x, y] for x in np.arange(0, 1, 0.01) for y in np.arange(0, 1, 0.01)])
Z = adaline.predict(xx)

fig = plt.figure()
ax = Axes3D(fig)
plt.title("Visualize the data")
plt.xlabel("x1")
plt.ylabel("x2")

ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], c="r", marker="o", zdir="z")
plt.show()

fig = plt.figure()
ax = Axes3D(fig)
plt.title("Visualize the data")
plt.xlabel("x1")
plt.ylabel("x2")

ax.plot(xx[:, 0], xx[:, 1], Z[:, 0], ".", zdir="z")
ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], c="r", marker="o")
plt.show()
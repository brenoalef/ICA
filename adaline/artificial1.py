import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from adaline import Adaline
import time


def f(x):
    return 3*x + 2 + np.random.rand()*3


dataset = np.array([[x, f(x)] for x in np.random.uniform(-30, 30, 100)])

mse = np.zeros((20, 1))
rmse = np.zeros((20, 1))
mean_time = 0
for i in range(20):
    X_train, X_test, Y_train, Y_test = train_test_split(dataset[:, 0], dataset[:, 1], test_size=0.80)
    X_train = X_train.reshape((X_train.shape[0], 1))
    X_test = X_test.reshape((X_test.shape[0], 1))
    Y_train = Y_train.reshape((Y_train.shape[0], 1))
    Y_test = Y_test.reshape((Y_test.shape[0], 1))
    
    start_time = time.clock()
    adaline = Adaline(n_iter=100)
    adaline.fit(X_train,Y_train)
    Y_hat = adaline.predict(X_test)
    mean_time += (time.clock() - start_time)/20    

    mse[i] = ((Y_test - Y_hat)**2).mean(axis=0)
    rmse[i] = mse[i]**(1./2)

print("Mean execution time", mean_time)
print("Standard Deviation (MSE)", np.std(mse, axis=0))
print("Standard Deviation (RMSE)",np.std(rmse, axis=0))

aa = np.linspace(np.min(X_test), np.max(X_test), num=90)
aa = aa.reshape((aa.shape[0], 1))
fig, ax = plt.subplots()
Z = adaline.predict(aa)
Z[Z >= 0] = 1
Z[Z != 1] = 0
plt.plot(aa, Z)
plt.plot(X_test[Y_test >= 0], Y_test[Y_test >= 0], "o", label="f(x) >= 0")
plt.plot(X_test[Y_test < 0], Y_test[Y_test < 0], "x", label="f(x) <= 0")
plt.title("Visualize the data")
leg = plt.legend(loc='lower right', ncol=2, shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from elm import ELM
import math
from mlxtend.plotting import plot_decision_regions


def f(x):
    return 2 * math.sin(x) + 3

dataset = np.array([[x, f(x)] for x in np.linspace(-100, 100, num=500)])

iters = 20
mse = np.zeros((iters, 1))
rmse = np.zeros((iters, 1))
for i in range(iters):
    X_train, X_test, Y_train, Y_test = train_test_split(dataset[:, :1], dataset[:, 1], test_size=0.33)
    Y_train = Y_train.reshape((Y_train.shape[0], 1))
    Y_test = Y_test.reshape((Y_test.shape[0], 1))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    elm = ELM(hidden_units=80)
    elm.fit(X_train,Y_train)
    Y_hat = elm.predict(X_test)

    mse[i] = ((Y_test - Y_hat)**2).mean(axis=0)
    rmse[i] = mse[i]**(1./2)

print("Average MSE", np.mean(mse, axis=0))
print("Average RMSE", np.mean(rmse, axis=0))
print("Standard Deviation (MSE)", np.std(mse, axis=0))
print("Standard Deviation (RMSE)",np.std(rmse, axis=0))

dataset[:, 0:1] = scaler.transform(dataset[:, 0:1])
fig, ax = plt.subplots()
Z = elm.predict(dataset[:, 0:1])
plt.plot(dataset[:, 0:1], Z, label="ELM output")
plt.plot(dataset[:, 0:1], dataset[:, 1], "-", label="Expected")
plt.title("Visualize the data")
leg = plt.legend(loc='lower right', ncol=2, shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)
plt.show()

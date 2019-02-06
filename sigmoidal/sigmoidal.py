import numpy as np
from sklearn.utils import shuffle


class Sigmoidal:
    def __init__(self, eta = 0.01, num_features = 3, n_iter = 2000, c=3, type="tan", tol=0.4):
        self.w = np.random.uniform(-1, 1, (c, num_features + 1))
        self.eta = eta
        self.n_iter = n_iter
        self.type=type
        self.tol = tol

    def __activation(self, u):
        if self.type == "log":
            return 1./(1 + np.exp(-u))
        else:
            return (1. - np.exp(-u))/(1 + np.exp(-u))
        
    def __updating_rule(self, w, error, y, x):
        if self.type == "log":
            return (w.T + self.eta * error * (y * (1. - y)) * x).T
        else:
            return (w.T + self.eta * error * (1./2 * (1 - y**2)) * x).T

    def fit(self, X, Y):
        X = np.hstack((np.ones((X.shape[0],1)), X))
        for n in range(self.n_iter):
            misses = 0
            X, Y = shuffle(X, Y)
            for i in range(len(X)):
                u = self.w.dot(X[i])
                y = self.__activation(u)
                if any(abs(y - Y[i]) > self.tol):
                    misses += 1
                    error = Y[i] - y
                    for c in range(self.w.shape[0]):
                        self.w[c] = self.__updating_rule(self.w[c], error[c], y[c], X[i])
            if misses == 0:
                break

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0],1)), X))
        Y_hat = X.dot(self.w.T)
        Y_hat = self.__activation(Y_hat)
        return Y_hat

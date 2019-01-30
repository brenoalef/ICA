import numpy as np


class Adaline:
    def __init__(self, eta = 0.0001, n_iter = 2000):
        self.eta = eta
        self.n_iter = n_iter
        self.error = []

    def fit(self, X, Y):
        X = np.hstack((np.ones((X.shape[0],1)), X))
        self.w  = np.random.uniform(-1, 1, (X.shape[1], 1))
        for n in range(self.n_iter):
            y = X.dot(self.w)
            error = Y - y
            self.w += self.eta * X.T.dot(error)
            cost = 1./2 * np.sum(error**2)
            self.error.append(cost)
        return self

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0],1)), X))
        Y_hat = X.dot(self.w)
        return Y_hat

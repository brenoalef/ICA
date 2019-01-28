import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class Sigmoidal:
    def __init__(self, eta = 0.01, num_features = 3, n_iter = 2000, c=3, type="tan"):
        self.w = np.zeros((c, num_features + 1))
        self.eta = eta
        self.n_iter = n_iter
        self.type=type

    def fit(self, X, Y):
        X = np.hstack((np.ones((X.shape[0],1)), X))
        for n in range(self.n_iter):
            misses = 0
            for i in range(len(X)):
                u = self.w.dot(X[i])
                if self.type == "log":
                    y = 1./(1 + np.exp(-u))
                else:
                    y = (1 - np.exp(-u))/(1 + np.exp(-u))
                if any(y != Y[i]):
                    misses += 1
                    error = Y[i] - y
                    for c in range(self.w.shape[0]):
                        if self.type == "log":
                            self.w[c] = (self.w[c].T + self.eta*error[c]*(y[c]*(1-y[c]))*X[i]).T
                        else:
                            self.w[c] = (self.w[c].T + self.eta*error[c]*(1./2*(1 - y[c]**2))*X[i]).T
            if misses == 0:
                break

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0],1)), X))
        Y_hat = X.dot(self.w.T)
        if self.type == "log":
            Y_hat = 1./(1 + np.exp(-Y_hat))
        else:
            Y_hat = (1 - np.exp(-Y_hat))/(1 + np.exp(-Y_hat))
        return Y_hat

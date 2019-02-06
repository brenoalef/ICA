import numpy as np
from sklearn.utils import shuffle

class SingleLayerPerceptron:
    def __init__(self, eta = 0.01, num_features = 3, n_iter = 2000, c=3):
        self.w = np.random.uniform(-1, 1, (c, num_features + 1))
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, Y):
        X = np.hstack((np.ones((X.shape[0],1)), X))
        for n in range(self.n_iter):
            misses = 0
            X, Y = shuffle(X, Y)
            for i in range(len(X)):
                u = self.w.dot(X[i])
                y = np.where(u >= 0, 1, 0)
                if any(y != Y[i]):
                    misses += 1
                    error = Y[i] - y
                    for c in range(self.w.shape[0]):
                        self.w[c] = (self.w[c].T + self.eta*error[c]*X[i]).T
            if misses == 0:
                break

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0],1)), X))
        Y_hat = X.dot(self.w.T)
        Y_hat = np.where(Y_hat >= 0, 1, 0)
        return Y_hat

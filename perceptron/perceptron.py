import numpy as np

class Perceptron:
    def __init__(self, eta = 0.01, n_iter = 2000):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, Y):
        X = np.hstack((np.ones((X.shape[0],1)), X))
        self.w  = np.random.uniform(-1, 1, (X.shape[1], 1))
        for n in range(self.n_iter):
            misses = 0
            for i in range(len(X)):
                u = np.dot(X[i], self.w)
                y = 1 if u >= 0 else 0
                if y != Y[i]:
                    misses += 1
                    error = Y[i] - y
                    self.w = (self.w.T + self.eta*error*X[i]).T
            if misses == 0:
                break

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0],1)), X))
        Y_hat = np.dot(X, self.w)
        Y_hat[Y_hat >= 0] = 1
        Y_hat[Y_hat != 1] = 0
        return Y_hat

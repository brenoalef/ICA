import numpy as np
from sklearn.utils import shuffle


class MLP:
    def __init__(self, hidden_units=5, activation="log", lr=0.01, n_iter=20):
        self.hidden_units = hidden_units
        self.activation = activation
        self.lr = lr
        self.n_iter = n_iter
    
    def init_weights(self, n_features, n_outputs):
        self.w = np.random.normal(size=(n_features + 1, self.hidden_units))
        self.m = np.random.normal(size=(self.hidden_units + 1, n_outputs))

    def __activation(self, u):
        if self.__activation == "log":
            return 1.0/(1.0 + np.exp(-u.astype(np.float)))
        else:
            return (1.0 - np.exp(-u.astype(np.float)))/(1.0 + np.exp(-u.astype(np.float)))

    def __activation_derivative(self, u):
        if self.activation == "log":
            return u * (1.0 - u)
        else:
            return (1.0/2.0) * (1.0 - np.square(u))

    def __forward(self, x):
        hu = x.dot(self.w)
        h = self.__activation(hu)
        h = np.hstack((np.ones((h.shape[0], 1)), h))
        u = h.dot(self.m)
        y = self.__activation(u)
        return y, h

    def __backprop(self, x, y):
        y_hat, h = self.__forward(x)
        error_y = y - y_hat
        delta_y = self.__activation_derivative(y_hat) * error_y
        m_aux = np.delete(self.m, np.s_[0], 0)
        h_aux = np.delete(h, np.s_[0], 1)
        error_h = delta_y.dot(m_aux.T)
        delta_h = self.__activation_derivative(h_aux) * error_h
        self.m += (self.lr * h.T.dot(delta_y))
        self.w += (self.lr * x.T.dot(delta_h))

    def fit(self, X, Y):
        self.init_weights(X.shape[1], Y.shape[1])
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        for _ in range(self.n_iter):
            X, Y = shuffle(X, Y)
            for i in range(X.shape[0]):
                self.__backprop(X[i:i+1], Y[i:i+1])
        return self

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        Y_hat, _ = self.__forward(X)
        return Y_hat    

import numpy as np

class ELM:
    def __init__(self, hidden_units=5, activation="log"):
        self.hidden_units = hidden_units
        self.activation = activation
    
    def init_weights(self, n_features):
        self.h = np.random.normal(size=(n_features, self.hidden_units))

    def __activation(self, X):
        if self.activation == "relu":
            return np.maximum(X, 0, X)
        elif self.__activation == "log":
            return 1.0/(1.0 + np.exp(-X.astype(float)))
        else:
            return (1.0 - np.exp(-X))/(1.0 + np.exp(-X))

    def forward(self, X):
        H = X.dot(self.h)
        H = self.__activation(H)
        return H

    def fit(self, X, Y):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.init_weights(X.shape[1])
        H = self.forward(X)
        H = np.hstack((np.ones((H.shape[0], 1)), H))
        self.w = np.linalg.pinv(H).dot(Y)
        return self

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        H = self.forward(X)
        H = np.hstack((np.ones((H.shape[0], 1)), H))
        return H.dot(self.w)    

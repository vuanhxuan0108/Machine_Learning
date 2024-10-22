import numpy as np


class LinearRegression:
    def __init__(self):
        self.w = None
        self.bias = None

    def fit(self, X, Y):
        X_bar = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        params = np.linalg.inv(X_bar.T @ X_bar) @ X_bar.T @ Y
        self.w = params[:-1]
        self.bias = params[-1] 

    def predict(self, X):
        X_bar = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        predicted_Y = X_bar @ np.append(self.w, self.bias)
        return predicted_Y



class StackingRegressor:
    def __init__(self, estimators = []):
        self.est = estimators
    def fit(self, X_Train, y_Train):
        for model in self.est:
            model.fit(X_Train.values, y_Train.values)
    def predict(self, X_Test):
        y_pred = 0
        for model in self.est:
            y_pred += model.predict(X_Test)
        return y_pred/len(self.est)
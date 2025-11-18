#%% Imports
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

#%% logreg class
class OVRLogRegClassifier:
    def __init__(self, C=1.0, solver='lbfgs', max_iter=1000, n_jobs=-1):
        base_model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver
        )
        self.model = OneVsRestClassifier(base_model, n_jobs=n_jobs)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

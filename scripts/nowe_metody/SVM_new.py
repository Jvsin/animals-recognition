from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

class SVMClassifier:
    def __init__(self, C=1.0, kernel="rbf", gamma="scale"):
        self.model = SVC(C=C, kernel=kernel, gamma=gamma)
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    def predict(self, X_test):
        return self.model.predict(X_test)
    def tune_parameters(self, X_train, y_train, cv=3):
        param_grid = {'C':[1, 3, 10, 30, 100], 'gamma': ['scale', 0.03, 0.01, 0.003],'kernel': ['rbf']}
        grid = GridSearchCV(SVC(),param_grid=param_grid,cv=cv,n_jobs=-1,scoring="accuracy",verbose=1)
        grid.fit(X_train, y_train)
        self.model = grid.best_estimator_
        
        return grid.best_params_, grid.best_score_

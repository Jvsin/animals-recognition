#%% Imports
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV

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
    
    def tune_parameters(self, X_train, y_train, param_grid=None):
        if param_grid is None:
            param_grid = {
                'estimator__C': [0.01, 0.1, 1, 10, 100],
                'estimator__solver': ['lbfgs', 'liblinear', 'sag', 'saga'],
                'estimator__max_iter': [500, 1000, 1500]
            }
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        
        return grid_search.best_params_, grid_search.best_score_

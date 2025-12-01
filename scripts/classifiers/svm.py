#%% Imports 
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import RandomizedSearchCV

#%% svm class 
class SVMClassifier:
    def __init__(self, C=1.0, kernel='rbf', gamma='scale'):
        base_model = SVC(C=C, kernel=kernel, gamma=gamma)
        self.model = OneVsRestClassifier(base_model)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def tune_parameters(self, X_train, y_train, param_grid=None, cv=5):
        if param_grid is None:
            param_grid = {
                'estimator__C': [0.1, 1, 10, 100],
                'estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'estimator__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
            }
        grid_search = RandomizedSearchCV(
            self.model, 
            param_grid, 
            cv=cv, 
            n_iter=10,
            scoring='accuracy',
            n_jobs=-1
            )
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        
        return grid_search.best_params_, grid_search.best_score_

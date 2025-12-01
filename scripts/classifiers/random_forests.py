#%% Imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

#%% Random Forest Classifier
class RFClassifier:
    def __init__(self, n_estimators=100, max_depth=None):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def tune_parameters(self, X_train, y_train, param_grid=None, cv=5):     
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
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
    
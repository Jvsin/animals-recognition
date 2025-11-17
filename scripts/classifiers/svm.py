#%% Imports 
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

#%% svm class 
class SVMClassifier:
    def __init__(self, C=1.0, kernel='rbf', gamma='scale'):
        base_model = SVC(C=C, kernel=kernel, gamma=gamma)
        self.model = OneVsRestClassifier(base_model)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

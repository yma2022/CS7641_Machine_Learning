# This is decision tree learner.
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import util


class KNeareatNeighbor(object):

    def __init__(self, dataset, verbose=False):
        """
        decision tree initialization
        """
        self.verbose = verbose
        self.clf = None
        self.dataset = dataset

    def train(self, X_train, y_train):
        """
        build and train decision tree
        """
        clf_knn = KNeighborsClassifier()
        param_grid = {'n_neighbors': np.arange(1, 30),
                      'weights': ['uniform', 'distance']}
        clf_knn.fit(X_train,y_train)
        print("KNN Accuracy on the Train set: ", clf_knn.score(X_train, y_train))

        grid_search = GridSearchCV(estimator=clf_knn, param_grid=param_grid, cv=4)
        
        grid_search.fit(X_train, y_train)
        best_knn = grid_search.best_estimator_
        print("Best parameters:", grid_search.best_params_)

        if self.verbose:
            util.plot_learning_curve(self.dataset, estimator=best_knn, X_train=X_train, y_train=y_train, param_range=np.linspace(0.1,1,5), title="Learning Curve for KNN")
            util.plot_validation_curve(self.dataset, best_knn, X_train, y_train, title="Validation Curve for KNN over n_neighbors", xlabel='n_neighbors', p_name="n_neighbors", p_range=np.arange(1, 30, 2), cv=4)
            util.plot_validation_curve(self.dataset, best_knn, X_train, y_train, title="Validation Curve for KNN over weights", xlabel='weights', p_name="weights", p_range=['uniform', 'distance'], cv=4)
        
        best_knn.fit(X_train, y_train)
        print("Best KNN Accuracy on the Train set: ", best_knn.score(X_train, y_train))
        self.clf = best_knn

    def query(self, X_test, y_test):
        y_pred = self.clf.predict(X_test)
        test_score = accuracy_score(y_test, y_pred)
        print("KNN Accuracy on the Test set: ", test_score)

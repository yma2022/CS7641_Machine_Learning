# This is decision tree learner.
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt


class KNeareatNeighbor(object):

    def __init__(self, verbose=False):
        """
        decision tree initialization
        """

    def build_tree(self, X_train, y_train):
        """
        build and train decision tree
        """
        clf_knn = KNeighborsClassifier(n_neighbors=10)
        param_grid = {'n_neighbors': range(1, 16),
                      'weights': ['uniform', 'distance']}
        clf_knn.fit(X_train,y_train)


        grid_search = GridSearchCV(estimator=clf_knn,
                                   param_grid=param_grid,
                                   scoring='accuracy',
                                   return_train_score=True,
                                   cv=4,
                                   verbose=1,
                                   n_jobs=-1,)
        
        grid_search.fit(X_train, y_train)
        best_knn = grid_search.best_estimator_
        print("Best parameters:", grid_search.best_params_)
        best_knn.fit(X_train, y_train)

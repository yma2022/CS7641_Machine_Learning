# This is decision tree learner.
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt


class SupportVectorMachine(object):

    def __init__(self, verbose=False):
        """
        decision tree initialization
        """
        self.verbose = verbose

    def build_tree(self, X_train, y_train):
        """
        build and train decision tree
        """
        clf_svm = SVC(C=5, gamma=0.05, kernel='linear', random_state=0)
        linear_param_grid = {'C': np.logspace(-3, 2, 6),
                             'max_iter': np.logspace(1, 4, 4)}
        clf_svm.fit(X_train,y_train)
        acc = clf_svm.score(X_train,y_train)
        print("Accuracy of Training Data:"+ str(acc))


        # linear_grid_search = GridSearchCV(estimator=clf_svm,
        #                                   param_grid=linear_param_grid,
        #                                   scoring='accuracy',
        #                                   return_train_score=True,
        #                                   cv=4,
        #                                   verbose=10,
        #                                   n_jobs=-1,)
        
        # linear_grid_search.fit(X_train, y_train)
        # best_svm = linear_grid_search.best_estimator_
        # print("Best parameters:", linear_grid_search.best_params_)
        # best_svm.fit(X_train, y_train)

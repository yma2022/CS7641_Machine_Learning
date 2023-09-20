# This is SVM learner.
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import util


class SupportVectorMachine(object):

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
        clf_svm = SVC(random_state=42)
        param_grid = {'C': np.logspace(-3, 2, 6),
                      'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
        clf_svm.fit(X_train,y_train)
        print("SVM Accuracy on the Train set: ", clf_svm.score(X_train, y_train))

        grid_search = GridSearchCV(estimator=clf_svm, param_grid=param_grid, cv=4)

        grid_search.fit(X_train, y_train)
        best_svm = grid_search.best_estimator_
        print("Best parameters:", grid_search.best_params_)
        best_svm.fit(X_train, y_train)
        print("Best SVM Accuracy on the Train set: ", best_svm.score(X_train, y_train))
        self.clf = best_svm


        if self.verbose:
            util.plot_learning_curve(self.dataset, clf_svm, X_train, y_train, np.linspace(.1, 1.0, 5), title="Learning Curve for SVM")
            util.plot_validation_curve(self.dataset, clf_svm, X_train, y_train, title="Validation Curve for SVM over C", xlabel='C', p_name="C", p_range=np.logspace(-3, 2, 6), cv=4, log = True)

    def query(self, X_test, y_test):
        y_pred = self.clf.predict(X_test)
        test_score = accuracy_score(y_test, y_pred)
        print("SVM Accuracy on the Test set: ", test_score)
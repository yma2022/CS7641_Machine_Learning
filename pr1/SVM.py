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
import time


class SupportVectorMachine(object):

    def __init__(self, dataset, verbose=False):
        """
        decision tree initialization
        """
        self.verbose = verbose
        self.clf = []
        self.dataset = dataset

    def train(self, X_train, y_train):
        """
        build and train decision tree
        """
        train_scores = []
        train_time = []
        best_scores = []
        for k in ['poly', 'rbf', 'sigmoid']:
            clf_svm = SVC(kernel=k, random_state=42)
            param_grid = {'C': np.logspace(-10, 5, 10),
                        'gamma': np.logspace(-10, 5, 10)}
            clf_svm.fit(X_train,y_train)
            train_scores.append(clf_svm.score(X_train, y_train))            

            grid_search = GridSearchCV(estimator=clf_svm, param_grid=param_grid, cv=4)

            grid_search.fit(X_train, y_train)
            best_svm = grid_search.best_estimator_
            print("Best parameters:", grid_search.best_params_)
            


            if self.verbose:
                util.svm_plot_learning_curve(self.dataset, best_svm, X_train, y_train, np.linspace(0.2,1,5, endpoint=True), title="Learning Curve for SVM", kernel=k)
                util.svm_plot_validation_curve(self.dataset, best_svm, X_train, y_train, title="Validation Curve for SVM over C", xlabel='C', kernel=k, p_name="C", p_range=np.logspace(-10, 10, 10), ylim=(0, 1.01), cv=4, log = True)
                util.svm_plot_validation_curve(self.dataset, best_svm, X_train, y_train, title="Validation Curve for SVM over gamma", xlabel='gamma', kernel=k, p_name="gamma", p_range=np.logspace(-10, 10, 10), ylim=(0, 1.01), cv=4, log = True)

            start = time.time()
            best_svm.fit(X_train, y_train)
            end = time.time()
            train_time.append(end - start)
            best_scores.append(best_svm.score(X_train, y_train))           
            self.clf.append(best_svm)

        print("SVM Accuracy on the Train setwith linear, poly, rbf and sigmoid: ", train_scores)
        print("SVM Training time with linear, poly, rbf and sigmoid:", train_time)
        print("Best SVM Accuracy on the Train set with linear, poly, rbf and sigmoid: ", best_scores)

    def query(self, X_test, y_test):
        res = []
        for clf in self.clf:
            y_pred = clf.predict(X_test)
            test_score = accuracy_score(y_test, y_pred)
            res.append(test_score)
        print("SVM Accuracy on the Test set with linear, poly, rbf and sigmoid: ", res)
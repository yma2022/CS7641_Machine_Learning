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


class SupportVectorMachine(object):

    def __init__(self, verbose=False):
        """
        decision tree initialization
        """
        self.verbose = verbose
        self.clf = None

    def train(self, X_train, y_train):
        """
        build and train decision tree
        """
        clf_svm = SVC(C=5, gamma=0.05, kernel='linear', random_state=42)
        param_grid = {'C': np.logspace(-3, 2, 6),
                      'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
        clf_svm.fit(X_train,y_train)
        print("SVM Accuracy on the Train set: ", clf_svm.score(X_train, y_train))

        grid_search = GridSearchCV(estimator=clf_svm, param_grid=param_grid, cv=4)

        grid_search.fit(X_train, y_train)
        best_svm = grid_search.best_estimator_
        print("Best parameters:", grid_search.best_params_)
        best_svm.fit(X_train, y_train)
        print("Best KNN Accuracy on the Train set: ", best_svm.score(X_train, y_train))
        self.clf = best_svm


        if self.verbose:
            plt.figure()
            plt.xlabel("Training examples")
            plt.ylabel("Score")
            train_sizes, train_scores, test_scores = learning_curve(clf_svm, X_train, y_train, cv=4, n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5), verbose=0)
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

            plt.grid()
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, alpha=0.1,
                            color="r")
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, alpha=0.1, color="g")
            plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                    label="Training score")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                    label="Cross-validation score")

            plt.legend(loc="best")
            plt.show()

            plt.figure()
            param_range=np.logspace(-6, 2, 6)
            train_scores, valid_scores = validation_curve(clf_svm, X_train, y_train, param_name="C", param_range=param_range, cv=4)
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(valid_scores, axis=1)
            test_scores_std = np.std(valid_scores, axis=1)
            plt.title("Validation curve with C")
            plt.xlabel("C")
            plt.ylabel("Score")
            plt.fill_between(param_range, train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, alpha=0.1,
                            color="r")
            plt.fill_between(param_range, test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, alpha=0.1, color="g")
            
            plt.semilogx(param_range, train_scores_mean, 'o-', label="Training score",
                color="r")
            plt.semilogx(param_range, test_scores_mean, 'o-', label="Cross-validation score",
                color="g")

            plt.xticks(param_range)
            plt.legend(loc="best")
            plt.show()

    def query(self, X_test, y_test):
        y_pred = self.clf.predict(X_test)
        test_score = accuracy_score(y_test, y_pred)
        print("SVM Accuracy on the Test set: ", test_score)
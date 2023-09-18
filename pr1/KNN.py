# This is decision tree learner.
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt


class KNeareatNeighbor(object):

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
        clf_knn = KNeighborsClassifier(n_neighbors=10)
        param_grid = {'n_neighbors': np.arange(1, 30),
                      'weights': ['uniform', 'distance']}
        clf_knn.fit(X_train,y_train)
        print("KNN Accuracy on the Train set: ", clf_knn.score(X_train, y_train))

        grid_search = GridSearchCV(estimator=clf_knn, param_grid=param_grid)
        
        grid_search.fit(X_train, y_train)
        best_knn = grid_search.best_estimator_
        print("Best parameters:", grid_search.best_params_)
        best_knn.fit(X_train, y_train)
        print("Best KNN Accuracy on the Train set: ", best_knn.score(X_train, y_train))
        self.clf = best_knn

        if self.verbose:
            plt.figure()
            plt.xlabel("Training examples")
            plt.ylabel("Score")
            train_sizes, train_scores, test_scores = learning_curve(clf_knn, X_train, y_train, cv=4, n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5), verbose=0)
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
            param_range=np.arange(1, 30)
            train_scores, valid_scores = validation_curve(clf_knn, X_train, y_train, param_name="n_neighbors", param_range=param_range, cv=4)
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(valid_scores, axis=1)
            test_scores_std = np.std(valid_scores, axis=1)
            plt.title("Validation curve with max_depth")
            plt.xlabel("n_neighbors")
            plt.ylabel("Score")
            plt.fill_between(param_range, train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, alpha=0.1,
                            color="r")
            plt.fill_between(param_range, test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, alpha=0.1, color="g")
            
            plt.plot(param_range, train_scores_mean, 'o-', label="Training score",
                color="r")
            plt.plot(param_range, test_scores_mean, 'o-', label="Cross-validation score",
                color="g")

            plt.xticks(param_range)
            plt.legend(loc="best")
            plt.show()


    def query(self, X_test, y_test):
        y_pred = self.clf.predict(X_test)
        test_score = accuracy_score(y_test, y_pred)
        print("KNN Accuracy on the Test set: ", test_score)

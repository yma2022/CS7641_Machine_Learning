# AdaBoost
# This is decision tree learner.
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt


class BoostClassifier(object):

    def __init__(self, verbose=False, max_depth=20, min_samples_leaf=3, min_samples_split=3, ccp_alpha=0.0):
        self.verbose = verbose
        self.clf = None

    def train(self, X_train, y_train):
        base_clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=1, min_samples_split=5, ccp_alpha=0.0, random_state=42)
        base_clf.fit(X_train, y_train)
        print("At depth:", base_clf.tree_.max_depth)
        print("Base Decision Tree Accuracy on the Train set: ", base_clf.score(X_train, y_train))
        boost_clf = AdaBoostClassifier(base_clf, n_estimators=200, random_state=42)
        boost_clf.fit(X_train, y_train)
        print("Boost Decision Tree Accuracy on the Train set: ", boost_clf.score(X_train, y_train))
        self.clf = boost_clf
        

        if self.verbose:

            plt.figure()
            plt.xlabel("Training examples")
            plt.ylabel("Score")
            train_sizes, train_scores, test_scores = learning_curve(boost_clf, X_train, y_train, cv=4, n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5), verbose=0)
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
            param_range=np.arange(1, 200, 20)
            train_scores, valid_scores = validation_curve(boost_clf, X_train, y_train, param_name="n_estimators", param_range=param_range, cv=4)
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(valid_scores, axis=1)
            test_scores_std = np.std(valid_scores, axis=1)
            
            plt.title("Validation curve with number of estimators")
            plt.xlabel("n_estimators")
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
        print("Boost Decision Tree Accuracy on the Test set: ", test_score)

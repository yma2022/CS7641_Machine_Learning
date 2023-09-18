# This is decision tree learner.
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from matplotlib import pyplot as plt


class DecisionTree(object):

    def __init__(self, verbose=False, max_depth=30, min_samples_leaf=10, min_samples_split=5, ccp_alpha=5):
        """
        decision tree initialization
        """
        self.verbose = verbose
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.ccp_alpha = ccp_alpha
        self.clf = None

    def train(self, X_train, y_train):
        """
        build and train decision tree
        """
        clf_dt = DecisionTreeClassifier(random_state=42)
        clf_dt.fit(X_train, y_train)

        print("At depth:", clf_dt.tree_.max_depth)
        print("Decision Tree Accuracy on the Train set: ", clf_dt.score(X_train, y_train))
        

        params = {'max_depth': np.arange(1, self.max_depth+1),
         'min_samples_split': np.arange(2, self.min_samples_split+1),
         'min_samples_leaf': np.arange(1, self.min_samples_leaf+1),
         'ccp_alpha': np.linspace(0, 0.035, self.ccp_alpha)}

        gcv = GridSearchCV(estimator=clf_dt,param_grid=params)
        gcv.fit(X_train,y_train)
        best_dt = gcv.best_estimator_
        print("Best parameters:", gcv.best_params_)
        best_dt.fit(X_train, y_train)
        print("At depth:", best_dt.tree_.max_depth)
        print("Decision Tree Accuracy on the Train set (with pruning): ", best_dt.score(X_train, y_train))
        self.clf = best_dt

        if self.verbose:
            plt.figure()
            plt.xlabel("Training examples")
            plt.ylabel("Score")
            train_sizes, train_scores, test_scores = learning_curve(clf_dt, X_train, y_train, cv=4, n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5), verbose=0)
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
            param_range=np.arange(1, self.max_depth+1)
            train_scores, valid_scores = validation_curve(clf_dt, X_train, y_train, param_name="max_depth", param_range=np.arange(1, self.max_depth+1), cv=4)
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(valid_scores, axis=1)
            test_scores_std = np.std(valid_scores, axis=1)
            plt.title("Validation curve with max_depth")
            plt.xlabel("Max_depth")
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

            plt.figure()
            param_range=np.linspace(0, 0.035, self.ccp_alpha)
            train_scores, valid_scores = validation_curve(clf_dt, X_train, y_train, param_name="ccp_alpha", param_range=np.linspace(0, 0.035, self.ccp_alpha), cv=4)
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(valid_scores, axis=1)
            test_scores_std = np.std(valid_scores, axis=1)
            plt.title("Validation curve with ccp_alpha")
            plt.xlabel("ccp_alpha")
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
        print("Decision Tree Accuracy on the Test set: ", test_score)
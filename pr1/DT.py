# This is decision tree learner.
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import util


class DecisionTree(object):

    def __init__(self, dataset, verbose=False, max_depth=30, min_samples_leaf=10, min_samples_split=5, ccp_alpha=5):
        """
        decision tree initialization
        """
        self.verbose = verbose
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.ccp_alpha = ccp_alpha
        self.clf = None
        self.dataset = dataset

    def train(self, X_train, y_train):
        """
        build and train decision tree
        """
        clf_dt = DecisionTreeClassifier(random_state=42)
        clf_dt.fit(X_train, y_train)

        print("At depth:", clf_dt.tree_.max_depth)
        print("Decision Tree Accuracy on the Train set: ", clf_dt.score(X_train, y_train))
        self.clf = clf_dt

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

            util.plot_learning_curve(self.dataset, estimator=clf_dt, X_train=X_train, y_train=y_train, param_range=np.linspace(0.1,1,5), title="Learning Curve for Decision Tree")
            util.plot_validation_curve(self.dataset, clf_dt, X_train, y_train, title="Validation Curve for Decision Tree over max_depth", xlabel='max_depth', p_name="max_depth", p_range=np.arange(1, self.max_depth+1), cv=4)
            util.plot_validation_curve(self.dataset, clf_dt, X_train, y_train, title="Validation Curve for Decision Tree over min_samples_split", xlabel='min_sample_split', p_name="min_samples_split", p_range=np.arange(2, self.min_samples_split+1), cv=4)

    def query(self, X_test, y_test):
        y_pred = self.clf.predict(X_test)
        test_score = accuracy_score(y_test, y_pred)
        print("Decision Tree Accuracy on the Test set: ", test_score)
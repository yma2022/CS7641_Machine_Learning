# This is decision tree learner.
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import util


class DecisionTree(object):

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
        clf_dt = DecisionTreeClassifier(random_state=42)
        clf_dt.fit(X_train, y_train)

        print("At depth:", clf_dt.tree_.max_depth)
        print("Decision Tree Accuracy on the Train set: ", clf_dt.score(X_train, y_train))

        params = {'max_depth': np.arange(1, 20),
         'min_samples_leaf': np.arange(1, 20),
         'ccp_alpha': np.linspace(0, 0.035, 20)}

        gcv = GridSearchCV(estimator=clf_dt,param_grid=params, cv=4)
        gcv.fit(X_train,y_train)
        best_dt = gcv.best_estimator_
        print("Best parameters:", gcv.best_params_)


        if self.verbose:
            util.plot_learning_curve(self.dataset, estimator=best_dt, X_train=X_train, y_train=y_train, param_range=np.linspace(0.1,1,5), title="Learning Curve for Decision Tree")
            util.plot_validation_curve(self.dataset, best_dt, X_train, y_train, title="Validation Curve for Decision Tree over max_depth", xlabel='max_depth', p_name="max_depth", p_range=np.arange(1, 30, 2), cv=4)
            util.plot_validation_curve(self.dataset, best_dt, X_train, y_train, title="Validation Curve for Decision Tree over ccp_alpha", xlabel='ccp_alpha', p_name="ccp_alpha", p_range=np.linspace(0, 0.035, 10),  cv=4)
            util.plot_validation_curve(self.dataset, best_dt, X_train, y_train, title="Validation Curve for Decision Tree over min_samples_leaf", xlabel='min_samples_leaf', p_name="min_samples_leaf", p_range=np.arange(1, 100, 10), cv=4)
        
        best_dt.fit(X_train, y_train)
        print("At depth:", best_dt.tree_.max_depth)
        print("Best Decision Tree Accuracy on the Train set: ", best_dt.score(X_train, y_train))
        self.clf = best_dt

    def query(self, X_test, y_test):
        y_pred = self.clf.predict(X_test)
        test_score = accuracy_score(y_test, y_pred)
        print("Decision Tree Accuracy on the Test set: ", test_score)
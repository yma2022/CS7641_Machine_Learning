# AdaBoost
# This is decision tree learner.
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import util
import time


class BoostClassifier(object):

    def __init__(self, dataset, verbose=False):
        self.verbose = verbose
        self.clf = None
        self.dataset = dataset

    def train(self, X_train, y_train):
        # set up a weak learner
        base_clf = DecisionTreeClassifier(max_depth=3, ccp_alpha=0.0005, random_state=42)
        base_clf.fit(X_train, y_train)
        print("At depth:", base_clf.tree_.max_depth)
        print("Base Decision Tree Accuracy on the Train set: ", base_clf.score(X_train, y_train))
        clf_boost = AdaBoostClassifier(base_clf, random_state=42)
        params = {'n_estimators': np.arange(1, 200, 20),
                  'learning_rate': np.logspace(-3, 1, 6)}

        gcv = GridSearchCV(estimator=clf_boost,param_grid=params, cv=4)
        gcv.fit(X_train,y_train)
        best_dt = gcv.best_estimator_
        print("Best parameters:", gcv.best_params_)
        

        if self.verbose:
            util.plot_learning_curve(self.dataset, estimator=best_dt, X_train=X_train, y_train=y_train, param_range=np.linspace(0.2,1,5, endpoint=True), title="Learning Curve for Boosting Decision Tree")
            util.plot_validation_curve(self.dataset, best_dt, X_train, y_train, title="Validation Curve for Boosting Decision Tree over n_estimators", xlabel='n_estimators', p_name="n_estimators", p_range=np.arange(1, 200, 20), cv=4)
            util.plot_validation_curve(self.dataset, best_dt, X_train, y_train, title="Validation Curve for Boosting Decision Tree over learning_rate", xlabel='learning_rate', p_name="learning_rate", p_range=np.logspace(-3, 1, 6), cv=4, log=True)

        start = time.time()
        best_dt.fit(X_train, y_train)
        end = time.time()
        print("Boost Decision Tree Training time:", end - start)
        print("Boost Decision Tree Accuracy on the Train set: ", best_dt.score(X_train, y_train))
        self.clf = best_dt

    def query(self, X_test, y_test):
        y_pred = self.clf.predict(X_test)
        test_score = accuracy_score(y_test, y_pred)
        print("Boost Decision Tree Accuracy on the Test set: ", test_score)

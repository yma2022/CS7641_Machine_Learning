# AdaBoost
# This is decision tree learner.
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import util


class BoostClassifier(object):

    def __init__(self, dataset, verbose=False):
        self.verbose = verbose
        self.clf = None
        self.dataset = dataset

    def train(self, X_train, y_train):
        # set up a weak learner
        base_clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=1, min_samples_split=5, ccp_alpha=0.0, random_state=42)
        base_clf.fit(X_train, y_train)
        print("At depth:", base_clf.tree_.max_depth)
        print("Base Decision Tree Accuracy on the Train set: ", base_clf.score(X_train, y_train))
        boost_clf = AdaBoostClassifier(base_clf, n_estimators=200, random_state=42)
        boost_clf.fit(X_train, y_train)
        print("Boost Decision Tree Accuracy on the Train set: ", boost_clf.score(X_train, y_train))
        self.clf = boost_clf
        

        if self.verbose:
            util.plot_learning_curve(self.dataset, estimator=boost_clf, X_train=X_train, y_train=y_train, param_range=np.linspace(0.1,1,5), title="Learning Curve for Boosting Decision Tree")
            util.plot_validation_curve(self.dataset, boost_clf, X_train, y_train, title="Validation Curve for Boosting Decision Tree over n_estimators", xlabel='n_estimators', p_name="n_estimators", p_range=np.arange(1, 200, 20), cv=4)

    def query(self, X_test, y_test):
        y_pred = self.clf.predict(X_test)
        test_score = accuracy_score(y_test, y_pred)
        print("Boost Decision Tree Accuracy on the Test set: ", test_score)

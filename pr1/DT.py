# This is decision tree learner.
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt


class DecisionTree(object):

    def __init__(self, verbose=False, max_depth=20, min_samples_leaf=3, min_samples_split=3, ccp_alpha=0.0):
        """
        decision tree initialization
        """
        self.verbose = verbose
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.ccp_alpha = ccp_alpha

    def build_tree(self, X_train, y_train):
        """
        build and train decision tree
        """
        SEED = 21
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.2, random_state=SEED)
        unpruned = DecisionTreeClassifier(max_depth=None)
        unpruned.fit(X_train, y_train)

        print("At depth:", unpruned.tree_.max_depth)
        print("Decision Tree Accuracy on the Train set (without pruning): ", unpruned.score(X_train, y_train))
        print("Decision Tree Accuracy on the Dev set(without pruning):", unpruned.score(X_val, y_val))

        params = {'max_depth': range(1, self.max_depth),
         'min_samples_split': range(2, self.min_samples_split),
         'min_samples_leaf': range(1, self.min_samples_leaf)}

        tree = DecisionTreeClassifier()
        gcv = GridSearchCV(estimator=tree,param_grid=params)
        gcv.fit(X_train,y_train)
        prepruned = gcv.best_estimator_
        print("Best parameters:", gcv.best_params_)
        prepruned.fit(X_train, y_train)
        print("At depth:", prepruned.tree_.max_depth)
        print("Decision Tree Accuracy on the Train set (with prepruning): ", prepruned.score(X_train, y_train))
        print("Decision Tree Accuracy on the Dev set(with prepruning):", prepruned.score(X_val, y_val))


        path = tree.cost_complexity_pruning_path(X_train, y_train)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        print(ccp_alphas)

        # For each alpha we will append our model to a list
        trees = []
        for ccp_alpha in ccp_alphas:
            tree = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
            tree.fit(X_train, y_train)
            trees.append(tree)

        trees = trees[:-1]
        ccp_alphas = ccp_alphas[:-1]
        node_counts = [tree.tree_.node_count for tree in trees]
        depth = [tree.tree_.max_depth for tree in trees]
        if self.verbose:
            plt.scatter(ccp_alphas,node_counts)
            plt.scatter(ccp_alphas,depth)
            plt.plot(ccp_alphas,node_counts,label='no of nodes',drawstyle="steps-post")
            plt.plot(ccp_alphas,depth,label='depth',drawstyle="steps-post")
            plt.legend()
            plt.show()
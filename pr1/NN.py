import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import util
import time


class NeuralNetwork(object):

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
        clf_nn = MLPClassifier(random_state=42, max_iter=2000)
        param_grid = {'hidden_layer_sizes': np.arange(1, 100, 10)}
        clf_nn.fit(X_train,y_train)
        print("NN Accuracy on the Train set: ", clf_nn.score(X_train, y_train))

        grid_search = GridSearchCV(estimator=clf_nn, param_grid=param_grid, cv=4)

        grid_search.fit(X_train, y_train)
        best_nn = grid_search.best_estimator_
        print("Best parameters:", grid_search.best_params_)

        plt.figure()
        plt.title("Loss curve")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.plot(best_nn.loss_curve_)
        plt.savefig("img/" + self.dataset + "_NN_loss_curve.png")
        plt.close()
        
        if self.verbose:
            util.plot_learning_curve(self.dataset, best_nn, X_train, y_train, np.linspace(0.2,1,5, endpoint=True), title="Learning Curve for NN")
            util.plot_validation_curve(self.dataset, best_nn, X_train, y_train, title="Validation Curve for Neural Network over hidden_layer_sizes", xlabel='hidden_layer_sizes', p_name="hidden_layer_sizes", p_range=np.arange(1, 100, 10), cv=4)
            util.plot_validation_curve(self.dataset, best_nn, X_train, y_train, title="Validation Curve for Neural Network over activation", xlabel='activation', p_name="activation", p_range=['identity', 'logistic', 'tanh', 'relu'], cv=4)
        
        start = time.time()
        best_nn.fit(X_train, y_train)
        end = time.time()
        print("NN Training time:", end - start)
        print("Best NN Accuracy on the Train set: ", best_nn.score(X_train, y_train))
        self.clf = best_nn

    def query(self, X_test, y_test):
        y_pred = self.clf.predict(X_test)
        test_score = accuracy_score(y_test, y_pred)
        print("NN Accuracy on the Test set: ", test_score)
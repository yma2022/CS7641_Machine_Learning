# running experiments

import numpy as np
import pandas as pd
import DT as dt
import KNN as knn
import time
import NN as nn
import SVM as svm
import Boosting as boost
from os.path import join
from sklearn.model_selection import train_test_split
from sklearn import datasets
def main():
    def run_experiment1(dt_flag=False, svm_flag=False, boost_flag=False, knn_flag=False, nn_flag=False):
        if dt_flag:
            dtlearner = dt.DecisionTree(dataset='digits', verbose=True)
            dtlearner.train(x_train, y_train)
            dtlearner.query(x_test, y_test)
        if boost_flag:
            boostlearner = boost.BoostClassifier(dataset='digits', verbose=True)
            boostlearner.train(x_train, y_train)
            boostlearner.query(x_test, y_test)
        if knn_flag:
            knnlearner = knn.KNeareatNeighbor(dataset='digits', verbose=True)
            knnlearner.train(x_train, y_train)
            knnlearner.query(x_test, y_test)
        if svm_flag:
            svmlearner = svm.SupportVectorMachine(dataset='digits', verbose=True)
            svmlearner.train(x_train, y_train)
            svmlearner.query(x_test, y_test)
        if nn_flag:
            nnlearner = nn.NeuralNetwork(dataset='digits', verbose=True)
            nnlearner.train(x_train, y_train)
            nnlearner.query(x_test, y_test)
    #
    # Load MINST dataset
    #
    np.random.seed(902764819)
    print('Loading MNIST dataset...')
    digits = datasets.load_digits()

    x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.20, random_state=42)
    print('MNIST dataset loaded.')
    
    run_experiment1(dt_flag=True, svm_flag=True, boost_flag=True, knn_flag=True, nn_flag=True)

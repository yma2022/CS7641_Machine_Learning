# running experiments

import numpy as np
import pandas as pd
import DT as dt
import matplotlib.pyplot as plt
import KNN as knn
import time
import NN as nn
import SVM as svm
import Boosting as boost
from os.path import join
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import datasets

def main():
    def run_experiment2(dt_flag=False, svm_flag=False, boost_flag=False, knn_flag=False, nn_flag=False):
        if dt_flag:
            dtlearner = dt.DecisionTree(dataset='cancer', verbose=True)
            dtlearner.train(x_train, y_train)
            dtlearner.query(x_test, y_test)
        if boost_flag:
            boostlearner = boost.BoostClassifier(dataset='cancer', verbose=True)
            boostlearner.train(x_train, y_train)
            boostlearner.query(x_test, y_test)
        if knn_flag:
            knnlearner = knn.KNeareatNeighbor(dataset='cancer', verbose=True)
            knnlearner.train(x_train, y_train)
            knnlearner.query(x_test, y_test)
        if svm_flag:
            svmlearner = svm.SupportVectorMachine(dataset='cancer', verbose=True)
            svmlearner.train(x_train, y_train)
            svmlearner.query(x_test, y_test)
        if nn_flag:
            nnlearner = nn.NeuralNetwork(dataset='cancer', verbose=True)
            nnlearner.train(x_train, y_train)
            nnlearner.query(x_test, y_test)
    #
    # Load breast cancer wisconsin dataset
    #
    np.random.seed(902764819)
    print('Loading Breast Cancer Wisconsin dataset...')
    data = pd.read_csv('data/breast_cancer_wisconsin.csv')
    data.drop(columns=['id', 'Unnamed: 32'], axis=1, inplace=True)
    x = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    y[y == 'M'] = 1
    y[y == 'B'] = 0
    y = y.astype(int)

    scaler = preprocessing.StandardScaler()
    x = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
    print('Breast Cancer Wisconsin dataset loaded.')

    
    x = preprocessing.scale(x)
    run_experiment2(dt_flag=True, svm_flag=True, boost_flag=True, knn_flag=True, nn_flag=True)
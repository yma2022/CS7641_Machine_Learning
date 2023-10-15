import numpy as np
import pandas as pd
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import mlrose_hiive
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing, datasets
import time
import util




if __name__ == "__main__":
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
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
    print('Breast Cancer Wisconsin dataset loaded.')
    train_data = []
    test_data = []
    mse_data = []
    train_time = []
    test_time = []
    iterations = [5, 50, 100, 200, 500, 1000, 2000]
    """
    Back Prop Case
    """
    train_acc_bp = []
    test_acc_bp = []
    mse_bp = []
    train_time_bp = []
    test_time_bp = []    
    for i in iterations:
        nn_model_bp = mlrose_hiive.NeuralNetwork(hidden_nodes = [20], activation ='relu', 
                                    algorithm ='gradient_descent', 
                                    max_iters = i, bias = True, is_classifier = True, 
                                    learning_rate = 1e-03, early_stopping = True, 
                                    max_attempts = 100, random_state = 42)
        train_start = time.time()
        nn_model_bp.fit(X_train, y_train)
        train_end = time.time()
        train_time_bp.append(train_end - train_start)
        # Predict labels for train set and assess accuracy
        y_train_pred_bp = nn_model_bp.predict(X_train)
        y_train_accuracy_bp = accuracy_score(y_train, y_train_pred_bp)
        train_acc_bp.append(y_train_accuracy_bp)
        # Predict labels for test set and assess accuracy
        test_start = time.time()
        y_test_pred_bp = nn_model_bp.predict(X_test)
        test_end = time.time()
        test_time_bp.append(test_end - test_start)
        y_test_accuracy_bp = accuracy_score(y_test, y_test_pred_bp)
        test_acc_bp.append(y_test_accuracy_bp)
        mse_bp.append(nn_model_bp.loss)
    train_data.append(train_acc_bp)
    test_data.append(test_acc_bp)
    train_time.append(train_time_bp)
    test_time.append(test_time_bp)
    mse_data.append(mse_bp)


    """
    Random Hill Climb Case
    """
    train_acc_rhc = []
    test_acc_rhc = []
    mse_rhc = []
    train_time_rhc = []
    test_time_rhc = []
    for i in iterations:
        nn_model_rhc = mlrose_hiive.NeuralNetwork(hidden_nodes = [20], activation ='relu', 
                                    algorithm ='random_hill_climb', 
                                    max_iters = i, bias = True, is_classifier = True, 
                                    learning_rate = 1e-03, early_stopping = True, 
                                    max_attempts = 100, random_state = 42)
        train_start = time.time()
        nn_model_rhc.fit(X_train, y_train)
        train_end = time.time()
        train_time_rhc.append(train_end - train_start)
        # Predict labels for train set and assess accuracy
        y_train_pred_rhc = nn_model_rhc.predict(X_train)
        y_train_accuracy_rhc = accuracy_score(y_train, y_train_pred_rhc)
        train_acc_rhc.append(y_train_accuracy_rhc)
        # Predict labels for test set and assess accuracy
        test_start = time.time()
        y_test_pred_rhc = nn_model_rhc.predict(X_test)
        test_end = time.time()
        test_time_rhc.append(test_end - test_start)
        y_test_accuracy_rhc = accuracy_score(y_test, y_test_pred_rhc)
        test_acc_rhc.append(y_test_accuracy_rhc)
        mse_rhc.append(nn_model_rhc.loss)

    train_data.append(train_acc_rhc)
    test_data.append(test_acc_rhc)
    train_time.append(train_time_rhc)
    test_time.append(test_time_rhc)
    mse_data.append(mse_rhc)



    """
    Simulated Annealing Case
    """
    train_acc_sa = []
    test_acc_sa = []
    mse_sa = []
    train_time_sa = []
    test_time_sa = []

    for i in iterations:
        nn_model_sa = mlrose_hiive.NeuralNetwork(hidden_nodes = [20], activation ='relu', 
                                    algorithm ='simulated_annealing', 
                                    max_iters = i, bias = True, is_classifier = True, 
                                    learning_rate = 1e-03, early_stopping = True, 
                                    max_attempts = 100, random_state = 42)
        train_start = time.time()
        nn_model_sa.fit(X_train, y_train)
        train_end = time.time()
        train_time_sa.append(train_end - train_start)
        # Predict labels for train set and assess accuracy
        y_train_pred_sa = nn_model_rhc.predict(X_train)
        y_train_accuracy_sa = accuracy_score(y_train, y_train_pred_sa)
        train_acc_sa.append(y_train_accuracy_sa)
        # Predict labels for test set and assess accuracy
        test_start = time.time()
        y_test_pred_sa = nn_model_sa.predict(X_test)
        test_end = time.time()
        test_time_sa.append(test_end - test_start)
        y_test_accuracy_sa = accuracy_score(y_test, y_test_pred_sa)
        test_acc_sa.append(y_test_accuracy_sa)
        mse_sa.append(nn_model_sa.loss)
    train_data.append(train_acc_sa)
    test_data.append(test_acc_sa)
    train_time.append(train_time_sa)
    test_time.append(test_time_sa)
    mse_data.append(mse_sa)

    """
    Genetic Algorithm Case
    """
    train_acc_ga = []
    test_acc_ga = []
    mse_ga = []
    train_time_ga = []
    test_time_ga = []
    for i in iterations:
        nn_model_ga = mlrose.NeuralNetwork(hidden_nodes = [20], activation ='relu', 
                                    algorithm ='genetic_alg', 
                                    max_iters = i, bias = True, is_classifier = True, 
                                    learning_rate = 1e-03, early_stopping = True, 
                                    max_attempts = 100, random_state = 42)
        train_start = time.time()
        nn_model_ga.fit(X_train, y_train)
        train_end = time.time()
        train_time_ga.append(train_end - train_start)
        # Predict labels for train set and assess accuracy
        y_train_pred_ga = nn_model_ga.predict(X_train)
        y_train_accuracy_ga = accuracy_score(y_train, y_train_pred_ga)
        train_acc_ga.append(y_train_accuracy_ga)
        # Predict labels for test set and assess accuracy
        test_start = time.time()
        y_test_pred_ga = nn_model_rhc.predict(X_test)
        test_end = time.time()
        test_time_ga.append(test_end - test_start)
        y_test_accuracy_ga = accuracy_score(y_test, y_test_pred_ga)
        test_acc_ga.append(y_test_accuracy_sa)
        mse_ga.append(nn_model_ga.loss)
    train_data.append(train_acc_ga)
    test_data.append(test_acc_ga)
    train_time.append(train_time_ga)
    test_time.append(test_time_ga)
    mse_data.append(mse_ga)

    print("iterations: ", iterations)
    print("train_data: ", train_data)
    print("test_data: ", test_data)
    print("train_time: ", train_time)
    print("test_time: ", test_time)
    util.plot_curve(train_data, iterations, ylabel="Training Accuracy", title="Breast Cancer Wisconsin Dataset - Neural Network Training Curve", show=False)
    util.plot_curve(test_data, iterations, ylabel="Testing Accuracy", title="Breast Cancer Wisconsin Dataset - Neural Network Testing Curve", show=False)
    util.plot_curve(train_time, iterations, ylabel="Training Time", title="Breast Cancer Wisconsin Dataset - Neural Network Training Time", show=False)
    util.plot_curve(test_time, iterations, ylabel="Testing Time", title="Breast Cancer Wisconsin Dataset - Neural Network Testing Time", show=False)
    util.plot_curve(mse_data, iterations, ylabel="MSE", title="Breast Cancer Wisconsin Dataset - Neural Network Mean Squared Error", show=False)
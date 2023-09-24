- [ ] Datasets
    - [ ] Select two datasets
    - [ ] Preprocess datasets
    - [ ] Choose a performance metric (accuracy, error, precision or recall) for each dataset and justify choice
    - [ ] Introduce each dataset and write why they are interesting, and brief description of the preprocessing
- [ ] Talk about overall experimental methodology
- [ ] Decision Trees (DT)
    - [ ] Learning Curve plot and analysis
        - [ ] Keep the optimal hyperparameter choice . Vary the training data set size , train models with data and plot the curves
        - [ ] bias and variance / overfit and underfit analysis
    - [ ] Validation curves for at least two hyperparameters
        - [ ] Pruning is a required hyperparameter
        - [ ] Vary hyperparameter of interest keeping all other optimal hyperparameters fixed and train the models and plot the curves 
        - [ ] bias and variance / overfit and underfit analysis
    - [ ] Add intermidate learning or validation curve to show suboptimal performance and analysis
    - [ ] Link back results to some algorithmic behavior, hyperparameter interaction between the algorithm and input data, etc
    - [ ] Document wall clock times
    - [ ] Tune model to be optimal (GridSearch allowed)
    - [ ] Analyze results isolated to other algoritms
    - [ ] Compare and contrast results across algorithms and across datasets
- [ ] Neural Networks (NN)
    - [ ] Learning Curve plot and analysis
        - [ ] X axis should be iterations or epoch - ie loss curve
        - [ ] Keep the optimal hyperparameter choice . Vary the training data set size , train models with data and plot the curves
        - [ ] Bias and variance / overfit and underfit analysis
    - [ ] Validation curves for at least two hyperparameters
        - [ ] Hidden Layer Size is a required hyperparameter
        - [ ] Vary hyperparameter of interest keeping all other optimal hyperparameters fixed and train the models and plot the curves 
        - [ ] Bias and variance / overfit and underfit analysis
    - [ ] Add intermidate learning or validation curve to show suboptimal performance and analysis
    - [ ] Link back results to some algorithmic behavior, hyperparameter interaction between the algorithm and input data, etc
    - [ ] Document wall clock times
    - [ ] Tune model to be optimal (GridSearch allowed)
    - [ ] Analyze results isolated to other algoritms
    - [ ] Compare and contrast results across algorithms and across datasets
- [ ] Boosting
    - [ ] Learning Curve plot and analysis
        - [ ] Keep the optimal hyperparameter choice . Vary the training data set size , train models with data and plot the curves
        - [ ] Bias and variance / overfit and underfit analysis
    - [ ] Validation curves for at least two hyperparameters
        - [ ] \# of Weak Learners is a required hyperparameter
        - [ ] Vary hyperparameter of interest keeping all other optimal hyperparameters fixed and train the models and plot the curves 
        - [ ] Bias and variance / overfit and underfit analysis
    - [ ] Add intermidate learning or validation curve to show suboptimal performance and analysis
    - [ ] Link back results to some algorithmic behavior, hyperparameter interaction between the algorithm and input data, etc
    - [ ] Document wall clock times
    - [ ] Tune model to be optimal (GridSearch allowed)
    - [ ] Analyze results isolated to other algoritms
    - [ ] Compare and contrast results across algorithms and across datasets
- [ ] Support vector machines (SVM)
    - [ ] Learning Curve plot and analysis
        - [ ] Keep the optimal hyperparameter choice . Vary the training data set size , train models with data and plot the curves
        - [ ] Bias and variance / overfit and underfit analysis
    - [ ] Validation curves for at least two hyperparameters
        - [ ] Kernel Type is a required hyperparameter
        - [ ] Vary hyperparameter of interest keeping all other optimal hyperparameters fixed and train the models and plot the curves 
        - [ ] Bias and variance / overfit and underfit analysis
    - [ ] Add intermidate learning or validation curve to show suboptimal performance and analysis
    - [ ] Link back results to some algorithmic behavior, hyperparameter interaction between the algorithm and input data, etc
    - [ ] Document wall clock times
    - [ ] Tune model to be optimal (GridSearch allowed)
    - [ ] Analyze results isolated to other algoritms
    - [ ] Compare and contrast results across algorithms and across datasets
- [ ] K-Nearest Neighbor (kNN)
    - [ ] Learning Curve plot and analysis
        - [ ] Keep the optimal hyperparameter choice . Vary the training data set size , train models with data and plot the curves
        - [ ] Bias and variance / overfit and underfit analysis
    - [ ] Validation curves for at least two hyperparameters
        - [ ] K is a required hyperparameter
        - [ ] Vary hyperparameter of interest keeping all other optimal hyperparameters fixed and train the models and plot the curves 
        - [ ] Bias and variance / overfit and underfit analysis
    - [ ] Add intermidate learning or validation curve to show suboptimal performance and analysis
    - [ ] Link back results to some algorithmic behavior, hyperparameter interaction between the algorithm and input data, etc
    - [ ] Document wall clock times
    - [ ] Tune model to be optimal (GridSearch allowed)
    - [ ] Analyze results isolated to other algoritms
    - [ ] Compare and contrast results across algorithms and across datasets
- [ ] Write solid conclusion

Loading MNIST dataset...
MNIST dataset loaded.
At depth: 15
Decision Tree Accuracy on the Train set:  1.0
Best parameters: {'ccp_alpha': 0.0, 'max_depth': 11}
Decision Tree Training time: 0.015166282653808594
Best Decision Tree Accuracy on the Train set:  0.9832985386221295
Decision Tree Accuracy on the Test set:  0.8388888888888889
At depth: 3
Base Decision Tree Accuracy on the Train set:  0.4732080723729993
Best parameters: {'learning_rate': 1.584893192461114, 'n_estimators': 161}
Boost Decision Tree Training time: 1.5463809967041016
Boost Decision Tree Accuracy on the Train set:  0.9749478079331941
Boost Decision Tree Accuracy on the Test set:  0.9583333333333334
KNN Accuracy on the Train set:  0.9895615866388309
Best parameters: {'n_neighbors': 4, 'weights': 'distance'}
KNN Training time: 0.0012154579162597656
Best KNN Accuracy on the Train set:  1.0
KNN Accuracy on the Test set:  0.9833333333333333
Best parameters: {'C': 9.999999999999999e-11, 'gamma': 46.41588833612792}
Best parameters: {'C': 46.41588833612792, 'gamma': 0.00046415888336127817}
Best parameters: {'C': 2154.4346900318865, 'gamma': 9.999999999999999e-06}
SVM Accuracy on the Train setwith linear, poly, rbf and sigmoid:  [0.9979123173277662, 0.9965205288796103, 0.9067501739735561]
SVM Training time with linear, poly, rbf and sigmoid: [0.02213001251220703, 0.04064059257507324, 0.04100751876831055]
Best SVM Accuracy on the Train set with linear, poly, rbf and sigmoid:  [1.0, 1.0, 1.0]
SVM Accuracy on the Test set with linear, poly, rbf and sigmoid:  [0.9861111111111112, 0.9888888888888889, 0.9777777777777777]
NN Accuracy on the Train set:  1.0
Best parameters: {'activation': 'logistic', 'hidden_layer_sizes': 71}
NN Training time: 1.6659457683563232
Best NN Accuracy on the Train set:  1.0
NN Accuracy on the Test set:  0.9805555555555555
Loading Breast Cancer Wisconsin dataset...
Breast Cancer Wisconsin dataset loaded.
At depth: 7
Decision Tree Accuracy on the Train set:  1.0
Best parameters: {'ccp_alpha': 0.00736842105263158, 'max_depth': 7}
Decision Tree Training time: 0.008636951446533203
Best Decision Tree Accuracy on the Train set:  0.989010989010989
Decision Tree Accuracy on the Test set:  0.956140350877193
At depth: 3
Base Decision Tree Accuracy on the Train set:  0.978021978021978
Best parameters: {'learning_rate': 1.584893192461114, 'n_estimators': 61}
Boost Decision Tree Training time: 0.38782358169555664
Boost Decision Tree Accuracy on the Train set:  1.0
Boost Decision Tree Accuracy on the Test set:  0.956140350877193
KNN Accuracy on the Train set:  0.9802197802197802
Best parameters: {'n_neighbors': 10, 'weights': 'distance'}
KNN Training time: 0.0010268688201904297
Best KNN Accuracy on the Train set:  1.0
KNN Accuracy on the Test set:  0.956140350877193
Best parameters: {'C': 4.641588833612772e-09, 'gamma': 46.41588833612792}
Best parameters: {'C': 46.41588833612792, 'gamma': 0.00046415888336127817}
Best parameters: {'C': 2154.4346900318865, 'gamma': 0.00046415888336127817}
SVM Accuracy on the Train setwith linear, poly, rbf and sigmoid:  [0.9142857142857143, 0.9868131868131869, 0.9516483516483516]
SVM Training time with linear, poly, rbf and sigmoid: [0.0032660961151123047, 0.003004312515258789, 0.0035986900329589844]
Best SVM Accuracy on the Train set with linear, poly, rbf and sigmoid:  [0.9736263736263736, 0.9824175824175824, 0.9868131868131869]
SVM Accuracy on the Test set with linear, poly, rbf and sigmoid:  [0.956140350877193, 0.9824561403508771, 0.956140350877193]
NN Accuracy on the Train set:  0.9956043956043956
Best parameters: {'activation': 'logistic', 'hidden_layer_sizes': 61}
NN Training time: 0.48192429542541504
Best NN Accuracy on the Train set:  0.9868131868131869
NN Accuracy on the Test set:  0.9912280701754386

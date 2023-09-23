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
Decision Tree Training time: 0.014246940612792969
Best Decision Tree Accuracy on the Train set:  0.9832985386221295
Decision Tree Accuracy on the Test set:  0.8388888888888889
At depth: 3
Base Decision Tree Accuracy on the Train set:  0.4732080723729993
Best parameters: {'learning_rate': 0.03162277660168379, 'n_estimators': 81}
Boost Decision Tree Training time: 0.6027874946594238
Boost Decision Tree Accuracy on the Train set:  0.9478079331941545
Boost Decision Tree Accuracy on the Test set:  0.9111111111111111
KNN Accuracy on the Train set:  0.9895615866388309
Best parameters: {'n_neighbors': 4, 'weights': 'distance'}
KNN Training time: 0.0020766258239746094
Best KNN Accuracy on the Train set:  1.0
KNN Accuracy on the Test set:  0.9833333333333333
SVM Accuracy on the Train set:  0.9965205288796103
Best parameters: {'C': 46.41588833612782, 'kernel': 'rbf'}
SVM Training time: 0.027467727661132812
Best SVM Accuracy on the Train set:  1.0
SVM Accuracy on the Test set:  0.9861111111111112
NN Accuracy on the Train set:  1.0
Best parameters: {'activation': 'logistic', 'hidden_layer_sizes': 21}
NN Training time: 1.8989779949188232
Best NN Accuracy on the Train set:  1.0
NN Accuracy on the Test set:  0.9638888888888889
Loading Breast Cancer Wisconsin dataset...
Breast Cancer Wisconsin dataset loaded.
At depth: 7
Decision Tree Accuracy on the Train set:  1.0
Best parameters: {'ccp_alpha': 0.00736842105263158, 'max_depth': 7}
Decision Tree Training time: 0.010214567184448242
Best Decision Tree Accuracy on the Train set:  0.989010989010989
Decision Tree Accuracy on the Test set:  0.956140350877193
At depth: 3
Base Decision Tree Accuracy on the Train set:  0.978021978021978
Best parameters: {'learning_rate': 1.0, 'n_estimators': 61}
Boost Decision Tree Training time: 0.2973771095275879
Boost Decision Tree Accuracy on the Train set:  1.0
Boost Decision Tree Accuracy on the Test set:  0.9649122807017544
KNN Accuracy on the Train set:  0.9802197802197802
Best parameters: {'n_neighbors': 10, 'weights': 'distance'}
KNN Training time: 0.0009984970092773438
Best KNN Accuracy on the Train set:  1.0
KNN Accuracy on the Test set:  0.956140350877193
SVM Accuracy on the Train set:  0.9868131868131869
Best parameters: {'C': 0.2782559402207126, 'kernel': 'linear'}
SVM Training time: 0.0039038658142089844
Best SVM Accuracy on the Train set:  0.9868131868131869
SVM Accuracy on the Test set:  0.9736842105263158
NN Accuracy on the Train set:  0.9956043956043956
Best parameters: {'activation': 'identity', 'hidden_layer_sizes': 6}
NN Training time: 0.15912890434265137
Best NN Accuracy on the Train set:  0.9846153846153847
NN Accuracy on the Test set:  0.9912280701754386

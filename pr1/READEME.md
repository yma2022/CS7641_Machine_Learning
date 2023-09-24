
Datasets:
1 - digits: The MNIST database (sklearn.datasets)
2 - cancer: Breast Cancer Wisconsin (https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

Environment:
python:  3.10.12
scikit-learn: 1.2.2
numpy: 1.24.2
matplotlib: 3.5.1

Code structure and running instruction: 
main.py
    - util.py
        plot functions
    - experiment1.py
        conduct experiment using digits data
    - experiment2.py
        conduct experiment using cancer data
    - supervised learner classes
        - DT.py
        - NN.py
        - Boosting.py
        - KNN.py
        - SVM.py
Run main.py with python3 main.py will generate all images under /img folder.
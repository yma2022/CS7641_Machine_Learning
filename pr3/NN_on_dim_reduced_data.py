import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import SparseRandomProjection
from sklearn.manifold import LocallyLinearEmbedding
from sklearn import preprocessing

import util
import time


# Load MINST dataset
#
# print('Loading MNIST dataset...')
# digits = datasets.load_digits()

# x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.20, random_state=42)
# x_train = x_train / 255.0
# x_test = x_test / 255.0
# print('MNIST dataset loaded.')
# clf_nn = MLPClassifier(hidden_layer_sizes=50, alpha=0.001, random_state=42, max_iter=2000)
# start = time.time()
# clf_nn.fit(x_train, y_train)
# end = time.time()
# print('Training time: ', end-start)
# print('Training accuracy: ', clf_nn.score(x_train, y_train))
# print('Test accuracy: ', clf_nn.score(x_test, y_test))

# clf = PCA(n_components=50, random_state=42)
# X_pca_train = clf.fit_transform(x_train)
# X_pca_test = clf.fit_transform(x_test)
# clf_nn = MLPClassifier(hidden_layer_sizes=50, alpha=0.001, random_state=42, max_iter=2000)
# start = time.time()
# clf_nn.fit(X_pca_train, y_train)
# end = time.time()
# print('PCA Training time: ', end-start)
# print('PCA Training accuracy: ', clf_nn.score(X_pca_train, y_train))
# print('PCA Test accuracy: ', clf_nn.score(X_pca_test, y_test))

# clf = FastICA(n_components=60, random_state=42)
# X_ica_train = clf.fit_transform(x_train)
# X_ica_test = clf.fit_transform(x_test)
# clf_nn = MLPClassifier(hidden_layer_sizes=50, alpha=0.001, random_state=42, max_iter=2000)
# start = time.time()
# clf_nn.fit(X_ica_train, y_train)
# end = time.time()
# print('ICA Training time: ', end-start)
# print('ICA Training accuracy: ', clf_nn.score(X_ica_train, y_train))
# print('ICA Test accuracy: ', clf_nn.score(X_ica_test, y_test))

# clf = SparseRandomProjection(n_components=65, random_state=42)
# X_rp_train = clf.fit_transform(x_train)
# X_rp_test = clf.fit_transform(x_test)
# clf_nn = MLPClassifier(hidden_layer_sizes=50, alpha=0.001, random_state=42, max_iter=2000)
# start = time.time()
# clf_nn.fit(X_rp_train, y_train)
# end = time.time()
# print('RP Training time: ', end-start)
# print('RP Training accuracy: ', clf_nn.score(X_rp_train, y_train))
# print('RP Test accuracy: ', clf_nn.score(X_rp_test, y_test))

# clf = LocallyLinearEmbedding(n_neighbors=30, n_components=60, method='standard', random_state=42)
# X_lle_train = clf.fit_transform(x_train)
# X_lle_test = clf.fit_transform(x_test)
# clf_nn = MLPClassifier(hidden_layer_sizes=50, alpha=0.001, random_state=42, max_iter=2000)
# start = time.time()
# clf_nn.fit(X_lle_train, y_train)
# end = time.time()
# print('LLE Training time: ', end-start)
# print('LLE Training accuracy: ', clf_nn.score(X_lle_train, y_train))
# print('LLE Test accuracy: ', clf_nn.score(X_lle_test, y_test))
####################################################################################################

print('Loading Breast Cancer Wisconsin dataset...')
data = pd.read_csv('data/breast_cancer_wisconsin.csv')
data.drop(columns=['id', 'Unnamed: 32'], axis=1, inplace=True)
X_cancer = data.iloc[:, 1:].values
y_cancer = data.iloc[:, 0].values
y_cancer[y_cancer == 'M'] = 1
y_cancer[y_cancer == 'B'] = 0
y_cancer = y_cancer.astype(int)

scaler = preprocessing.StandardScaler()
X_cancer = scaler.fit_transform(X_cancer)
print('Breast Cancer Wisconsin dataset loaded.')
x_train, x_test, y_train, y_test = train_test_split(X_cancer, y_cancer, test_size=0.20, random_state=42)

clf_nn = MLPClassifier(hidden_layer_sizes=50, alpha=0.001, random_state=42, max_iter=2000)
start = time.time()
clf_nn.fit(x_train, y_train)
end = time.time()
print('Training time: ', end-start)
print('Training accuracy: ', clf_nn.score(x_train, y_train))
print('Test accuracy: ', clf_nn.score(x_test, y_test))

clf = PCA(n_components=20, random_state=42)
X_pca_train = clf.fit_transform(x_train)
X_pca_test = clf.fit_transform(x_test)
clf_nn = MLPClassifier(hidden_layer_sizes=50, alpha=0.001, random_state=42, max_iter=2000)
start = time.time()
clf_nn.fit(X_pca_train, y_train)
end = time.time()
print('PCA Training time: ', end-start)
print('PCA Training accuracy: ', clf_nn.score(X_pca_train, y_train))
print('PCA Test accuracy: ', clf_nn.score(X_pca_test, y_test))

clf = FastICA(n_components=23, random_state=42)
X_ica_train = clf.fit_transform(x_train)
X_ica_test = clf.fit_transform(x_test)
clf_nn = MLPClassifier(hidden_layer_sizes=50, alpha=0.001, random_state=42, max_iter=2000)
start = time.time()
clf_nn.fit(X_ica_train, y_train)
end = time.time()
print('ICA Training time: ', end-start)
print('ICA Training accuracy: ', clf_nn.score(X_ica_train, y_train))
print('ICA Test accuracy: ', clf_nn.score(X_ica_test, y_test))

clf = SparseRandomProjection(n_components=30, random_state=42)
X_rp_train = clf.fit_transform(x_train)
X_rp_test = clf.fit_transform(x_test)
clf_nn = MLPClassifier(hidden_layer_sizes=50, alpha=0.001, random_state=42, max_iter=2000)
start = time.time()
clf_nn.fit(X_rp_train, y_train)
end = time.time()
print('RP Training time: ', end-start)
print('RP Training accuracy: ', clf_nn.score(X_rp_train, y_train))
print('RP Test accuracy: ', clf_nn.score(X_rp_test, y_test))

clf = LocallyLinearEmbedding(n_neighbors=30, n_components=30, method='standard', random_state=42)
X_lle_train = clf.fit_transform(x_train)
X_lle_test = clf.fit_transform(x_test)
clf_nn = MLPClassifier(hidden_layer_sizes=50, alpha=0.001, random_state=42, max_iter=2000)
start = time.time()
clf_nn.fit(X_lle_train, y_train)
end = time.time()
print('LLE Training time: ', end-start)
print('LLE Training accuracy: ', clf_nn.score(X_lle_train, y_train))
print('LLE Test accuracy: ', clf_nn.score(X_lle_test, y_test))
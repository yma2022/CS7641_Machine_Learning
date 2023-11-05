import pandas as pd
import numpy as np
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn import datasets
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import SparseRandomProjection
from sklearn.manifold import LocallyLinearEmbedding
import collections
import time
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import util

'''
Expectation Maximization
k-Means
'''

    

print('Loading MNIST dataset...')
digits = datasets.load_digits()
X_digits = digits.data / 255.0
y_digits = digits.target
print('MNIST dataset loaded.')
X_red_p = PCA(n_components=50, random_state=42).fit_transform(X_digits)
X_red_l = LocallyLinearEmbedding(n_neighbors=30, n_components=60, random_state=42).fit_transform(X_digits)


kmodel = kmeans(n_clusters=10, random_state=42)
y_kmeans_0 = kmodel.fit_predict(X_digits)
y_kmeans_1 = kmodel.fit_predict(X_red_p)
y_kmeans_2 = kmodel.fit_predict(X_red_l)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('K-Means Clustering with Original, PCA and LLE Digits Data')
ax1.scatter(X_digits[:, 0], X_digits[:, 1], c=y_kmeans_0, s=10, cmap='viridis')
ax2.scatter(X_red_p[:, 0], X_red_p[:, 1], c=y_kmeans_1, s=10, cmap='viridis')
ax3.scatter(X_red_l[:, 0], X_red_l[:, 1], c=y_kmeans_2, s=10, cmap='viridis')
plt.savefig('./img/kmeans_digits_reduced.png')
plt.close()


emodel = GMM(n_components=10, random_state=42)
y_em_0 = emodel.fit_predict(X_digits)
y_em_1 = emodel.fit_predict(X_red_p)
y_em_2 = emodel.fit_predict(X_red_l)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('EM Clustering with Original, PCA and LLE Digits Data')
ax1.scatter(X_digits[:, 0], X_digits[:, 1], c=y_em_0, s=10, cmap='viridis')
ax2.scatter(X_red_p[:, 0], X_red_p[:, 1], c=y_em_1, s=10, cmap='viridis')
ax3.scatter(X_red_l[:, 0], X_red_l[:, 1], c=y_em_2, s=10, cmap='viridis')
plt.savefig('./img/em_digits_reduced.png')
plt.close()

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


X_red_p = PCA(n_components=20, random_state=42).fit_transform(X_cancer)
X_red_l = LocallyLinearEmbedding(n_neighbors=30, n_components=30, random_state=42).fit_transform(X_cancer)


kmodel = kmeans(n_clusters=2, random_state=42)
y_kmeans_0 = kmodel.fit_predict(X_cancer)
y_kmeans_1 = kmodel.fit_predict(X_red_p)
y_kmeans_2 = kmodel.fit_predict(X_red_l)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('K-Means Clustering with Original, PCA and LLE Cancer Data')
ax1.scatter(X_cancer[:, 0], X_cancer[:, 1], c=y_kmeans_0, s=10, cmap='viridis')
ax2.scatter(X_red_p[:, 0], X_red_p[:, 1], c=y_kmeans_1, s=10, cmap='viridis')
ax3.scatter(X_red_l[:, 0], X_red_l[:, 1], c=y_kmeans_2, s=10, cmap='viridis')
plt.savefig('./img/kmeans_cancer_reduced.png')
plt.close()


emodel = GMM(n_components=2, random_state=42)
y_em_0 = emodel.fit_predict(X_cancer)
y_em_1 = emodel.fit_predict(X_red_p)
y_em_2 = emodel.fit_predict(X_red_l)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('EM Clustering with Original, PCA and LLE Cancer Data')
ax1.scatter(X_cancer[:, 0], X_cancer[:, 1], c=y_em_0, s=10, cmap='viridis')
ax2.scatter(X_red_p[:, 0], X_red_p[:, 1], c=y_em_1, s=10, cmap='viridis')
ax3.scatter(X_red_l[:, 0], X_red_l[:, 1], c=y_em_2, s=10, cmap='viridis')
plt.savefig('./img/em_cancer_reduced.png')
plt.close()

            


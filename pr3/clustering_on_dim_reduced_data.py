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
import util

'''
Expectation Maximization
k-Means
'''


def clustering(X, y, method, metrics=defaultdict(list)):
    k_range=np.arange(2,150,10)
    if method == 'kmeans':
        for k in k_range:
            metrics['k'].append(k)
            model = kmeans(n_clusters=k, random_state=42)
            start = time.time()
            model.fit(X)
            end = time.time()            
            y_pred = model.predict(X)
            
            metrics['time'].append(end-start)
            metrics["SSE"].append(model.inertia_)
            metrics['Silhouette'].append(silhouette_score(X, y_pred))
            metrics['AMI'].append(ami(y, y_pred, average_method='max'))

    elif method == 'gmm':
        for k in k_range:
            metrics['k'].append(k)
            model = GMM(n_components=k, random_state=42)
            start = time.time()
            model.fit(X)
            end = time.time()
            y_pred = model.predict(X)
            
            metrics['time'].append(end-start)
            metrics["BIC"].append(model.bic(X))
            metrics["Log_Likelihood"].append(model.score(X))
            metrics['AMI'].append(ami(y, y_pred, average_method='max'))
        
    else:
        raise ValueError('Invalid method')
    

print('Loading MNIST dataset...')
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
print('MNIST dataset loaded.')
digits = collections.defaultdict(dict)
for f in ['PCA', 'ICA', 'RP', 'LLE']:
    digits_kmeans = collections.defaultdict(list)
    digits_gmm = collections.defaultdict(list)
    if f == 'PCA':
        X_red = PCA(n_components=10, random_state=42).fit_transform(X_digits)
    elif f == 'ICA':
        X_red = FastICA(n_components=10, random_state=42).fit_transform(X_digits)
    elif f == 'RP':
        X_red = SparseRandomProjection(n_components=10, random_state=42).fit_transform(X_digits)
    elif f == 'LLE':
        X_red = LocallyLinearEmbedding(n_neighbors=30, n_components=10, random_state=42).fit_transform(X_digits)
    clustering(X_red, y_digits, 'kmeans', digits_kmeans)
    digits[f]['K-Means'] = digits_kmeans
    
    clustering(X_red, y_digits, 'gmm', digits_gmm)
    digits[f]['GMM'] = digits_gmm
# print(digits)
for method in ['K-Means', 'GMM']:
    
    if method == 'K-Means':
        for param in ['SSE', 'Silhouette', 'AMI']:
            util.multiplot_curve(digits, method=method, param=param, dataset="Digits")
    else:
        for param in ['BIC', 'Log_Likelihood', 'AMI']:
            util.multiplot_curve(digits, method=method, param=param, dataset="Digits")


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
cancer = collections.defaultdict(dict)

for f in ['PCA', 'ICA', 'RP', 'LLE']:
    cancer_kmeans = collections.defaultdict(list)
    cancer_gmm = collections.defaultdict(list)
    if f == 'PCA':
        X_red = PCA(n_components=10, random_state=42).fit_transform(X_cancer)
    elif f == 'ICA':
        X_red = FastICA(n_components=10, random_state=42).fit_transform(X_cancer)
    elif f == 'RP':
        X_red = SparseRandomProjection(n_components=10, random_state=42).fit_transform(X_cancer)
    elif f == 'LLE':
        X_red = LocallyLinearEmbedding(n_neighbors=30, n_components=10, random_state=42).fit_transform(X_cancer)
    clustering(X_red, y_cancer, 'kmeans', cancer_kmeans)
    cancer[f]['K-Means'] = cancer_kmeans
    
    clustering(X_red, y_cancer, 'gmm', cancer_gmm)
    cancer[f]['GMM'] = cancer_gmm

for method in ['K-Means', 'GMM']:
    if method == 'K-Means':
        for param in ['SSE', 'Silhouette', 'AMI']:
            util.multiplot_curve(cancer, method=method, param=param, dataset="Cancer")
    else:
        for param in ['BIC', 'Log_Likelihood', 'AMI']:
            util.multiplot_curve(cancer, method=method, param=param, dataset="Cancer")
            


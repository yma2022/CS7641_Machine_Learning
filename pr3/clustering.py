import pandas as pd
import numpy as np
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn import datasets
from sklearn import preprocessing
import collections
import time
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
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
X_digits = digits.data / 255.0
y_digits = digits.target
print('MNIST dataset loaded.')

# kmodel = kmeans(n_clusters=10, random_state=42)
# y_kmeans = kmodel.fit_predict(X_digits)
# plt.figure()
# plt.scatter(X_digits[:, 0], X_digits[:, 1], c=y_kmeans, s=50, cmap='viridis')

# plt.title('K-Means Clustering')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.show()

# emodel = GMM(n_components=10, random_state=42)
# y_em = emodel.fit_predict(X_digits)
# plt.figure()
# plt.scatter(X_digits[:, 0], X_digits[:, 1], c=y_em, s=50, cmap='viridis')
# plt.title('EM Clustering')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.show()

digits_kmeans = collections.defaultdict(list)
digits_gmm = collections.defaultdict(list)
clustering(X_digits, y_digits, 'kmeans', digits_kmeans)

util.plot_curve(digits_kmeans, 'k', ['SSE', 'Silhouette', 'AMI'], 'K-Means metrics vs k Digits data', show=False)



clustering(X_digits, y_digits, 'gmm', digits_gmm)

util.plot_curve(digits_gmm, 'k', ['BIC', 'Log_Likelihood', 'AMI'], 'EM metrics vs k Digits data', show=False)



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

# kmodel = kmeans(n_clusters=2, random_state=42)
# y_kmeans = kmodel.fit_predict(X_cancer)
# plt.figure()
# plt.scatter(X_cancer[:, 0], X_cancer[:, 1], c=y_kmeans, s=50, cmap='viridis')

# plt.title('K-Means Clustering')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.show()

# emodel = GMM(n_components=2, random_state=42)
# y_em = emodel.fit_predict(X_cancer)
# plt.figure()
# plt.scatter(X_cancer[:, 0], X_cancer[:, 1], c=y_em, s=50, cmap='viridis')
# plt.title('EM Clustering')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.show()

cancer_kmeans = collections.defaultdict(list)
cancer_gmm = collections.defaultdict(list)
clustering(X_cancer, y_cancer, 'kmeans', cancer_kmeans)

util.plot_curve(cancer_kmeans, 'k', ['SSE', 'Silhouette', 'AMI'], 'K-Means metrics vs k Cancer data', show=False)

clustering(X_cancer, y_cancer, 'gmm', cancer_gmm)

util.plot_curve(cancer_gmm, 'k', ['BIC', 'Log_Likelihood', 'AMI'], 'EM metrics vs k Cancer data', show=False)

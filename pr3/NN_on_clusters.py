import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.model_selection import train_test_split
import time
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn import preprocessing

new_digits_kmeans = None
new_digits_gmm = None
new_cancer_kmeans = None
new_cancer_gmm = None

# Load MINST dataset
#
print('Loading MNIST dataset...')
digits = datasets.load_digits()

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.20, random_state=42)
X_digits = digits.data
y_digits = digits.target 
x_test = x_test
print('MNIST dataset loaded.')


model = kmeans(n_clusters=10, random_state=42)
model.fit(X_digits)
new_digits_kmeans = model.labels_.reshape(-1,1)

model = GMM(n_components=10, random_state=42)
model.fit(X_digits)
new_digits_gmm = model.predict(X_digits).reshape(-1,1)
  


#############################################################################################################

clf_nn = MLPClassifier(hidden_layer_sizes=50, alpha=0.001, random_state=42, max_iter=2000)
start = time.time()
clf_nn.fit(x_train, y_train)
end = time.time()
print('Training time: ', end-start)
print('Training accuracy: ', clf_nn.score(x_train, y_train))
print('Test accuracy: ', clf_nn.score(x_test, y_test))

x_train, x_test, y_train, y_test = train_test_split(new_digits_kmeans, y_digits, test_size=0.20, random_state=42)
clf_nn = MLPClassifier(hidden_layer_sizes=50, alpha=0.001, random_state=42, max_iter=2000)
start = time.time()
clf_nn.fit(x_train, y_train)
end = time.time()
print('K-Means Training time: ', end-start)
print('K-Means Training accuracy: ', clf_nn.score(x_train, y_train))
print('K-Means Test accuracy: ', clf_nn.score(x_test, y_test))

x_train, x_test, y_train, y_test = train_test_split(new_digits_gmm, y_digits, test_size=0.20, random_state=42)
clf_nn = MLPClassifier(hidden_layer_sizes=50, alpha=0.001, random_state=42, max_iter=2000)
start = time.time()
clf_nn.fit(x_train, y_train)
end = time.time()
print('GMM Training time: ', end-start)
print('GMM Training accuracy: ', clf_nn.score(x_train, y_train))
print('GMM Test accuracy: ', clf_nn.score(x_test, y_test))



#############################################################################################################


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
x_train, x_test, y_train, y_test = train_test_split(X_cancer, y_cancer, test_size=0.20, random_state=42)
print('Breast Cancer Wisconsin dataset loaded.')


model = kmeans(n_clusters=2, random_state=42)
model.fit(X_cancer)
new_cancer_kmeans = model.labels_.reshape(-1,1)

model = GMM(n_components=2, random_state=42)
model.fit(X_cancer)
new_cancer_gmm = model.predict(X_cancer).reshape(-1,1)
  


#############################################################################################################

clf_nn = MLPClassifier(hidden_layer_sizes=50, alpha=0.001, random_state=42, max_iter=2000)
start = time.time()
clf_nn.fit(x_train, y_train)
end = time.time()
print('Training time: ', end-start)
print('Training accuracy: ', clf_nn.score(x_train, y_train))
print('Test accuracy: ', clf_nn.score(x_test, y_test))

x_train, x_test, y_train, y_test = train_test_split(new_cancer_kmeans, y_cancer, test_size=0.20, random_state=42)
clf_nn = MLPClassifier(hidden_layer_sizes=50, alpha=0.001, random_state=42, max_iter=2000)
start = time.time()
clf_nn.fit(x_train, y_train)
end = time.time()
print('K-Means Training time: ', end-start)
print('K-Means Training accuracy: ', clf_nn.score(x_train, y_train))
print('K-Means Test accuracy: ', clf_nn.score(x_test, y_test))

x_train, x_test, y_train, y_test = train_test_split(new_cancer_gmm, y_cancer, test_size=0.20, random_state=42)
clf_nn = MLPClassifier(hidden_layer_sizes=50, alpha=0.001, random_state=42, max_iter=2000)
start = time.time()
clf_nn.fit(x_train, y_train)
end = time.time()
print('GMM Training time: ', end-start)
print('GMM Training accuracy: ', clf_nn.score(x_train, y_train))
print('GMM Test accuracy: ', clf_nn.score(x_test, y_test))
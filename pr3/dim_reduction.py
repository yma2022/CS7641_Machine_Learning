import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import SparseRandomProjection
from sklearn.manifold import LocallyLinearEmbedding
from sklearn import datasets
from scipy.stats import kurtosis
from sklearn import preprocessing
import util
nn_reg = [0.01,  0.0001]
nn_arch= [(50,50),(50,),(100,)]

# Scale and visualize the embedding vectors
def plot_embedding(dataset, X,y,vals, title=None):
    if dataset == 'Digits':
        data = X_digits
    elif dataset == 'Cancer':
        data = X_cancer
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()
    plt.xlabel('Dim1')
    plt.ylabel('Dim2')
    for i in range(data.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),color=plt.cm.Set1(y[i]/vals),fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    plt.title(title)
    plt.savefig('./img/'+title+'.png')
'''
Dimensionality Reduction
'''

'''
run PCA
'''
def run_PCA(dataset, X, y, title, k=10, plot=False):
    # Create a PCA instance: pca
    pca = PCA(random_state=5)
    pca.fit(X)
    tmp = pd.Series(data = pca.explained_variance_,index = range(1,pca.explained_variance_.shape[0]+1))
    tmp.to_csv('./outputs/PCA/'+title+'.csv')
    if plot:
        plt.figure()
        plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, 'o-')
        plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), 'o-')
        plt.title("Component-wise and Cumulative Explained Variance: "+title)
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.legend(['Component-wise', 'Cumulative'])
        plt.savefig('./img/'+'PCA_'+title+'.png')
        plt.close()

        X_pca = PCA(n_components=2, random_state=5).fit_transform(X)
        plot_embedding(dataset, X_pca,y,k, "PCA projection first 2 components: "+title)
        X_pca = pd.DataFrame(X_pca)
        X_pca.to_csv('./outputs/PCA/'+title+'_pca.csv')

'''
run ICA
'''
def run_ICA(dataset, X, y, title, k=10, dims=[]):
    X_ica = FastICA(n_components=2, random_state=42).fit_transform(X)
    plot_embedding(dataset, X_ica,y,k, "ICA projection first 2 components: "+title)
    clf = FastICA(random_state=42)
    kurt = {}
    for dim in dims:
        clf.set_params(n_components=dim)
        ica = clf.fit(X)
        components = ica.components_
        kurt[dim] = np.mean(kurtosis(components))

    kurt = pd.Series(kurt) 

    plt.figure()
    plt.plot(kurt.loc[dims], 'o-')
    plt.title("Measure of non-Gaussianity: "+title)
    plt.xlabel('number of components')
    plt.ylabel('Kurtosis')
    plt.savefig('./img/'+'ICA_'+title+'.png')

def run_RP(dataset, X, y, title, k=10, dims=[]):
    X_rp = SparseRandomProjection(n_components=2, random_state=5).fit_transform(X)
    plot_embedding(dataset, X_rp,y,k, "RP projection first 2 components: "+title)

    clf = SparseRandomProjection(random_state=5)
    reconstruction_error = {}
    for dim in dims:
        clf.set_params(n_components=dim)
        tmp = clf.fit_transform(X)
        tmp = pd.DataFrame(tmp)
        tmp = np.linalg.norm(X - clf.inverse_transform(tmp), axis=1)
        reconstruction_error[dim] = np.mean(tmp)

    reconstruction_error = pd.Series(reconstruction_error) 

    plt.figure()    
    plt.plot(reconstruction_error.loc[dims], 'o-')
    plt.title('Reconstruction Error vs. Number of Components: '+title)
    plt.xlabel('number of components')
    plt.ylabel('reconstruction error')
    plt.savefig('./img/'+'RP_'+title+'.png')


def run_LLE(dataset, X, y, title, k=10, dims=[]):
    clf = LocallyLinearEmbedding(n_neighbors=30, n_components=2, method='standard', random_state=42)
    X_lle = clf.fit_transform(X)
    plot_embedding(dataset, X_lle,y,k, "LLE projection first 2 components: "+title)
    reconstruction_error = {}
    for dim in dims:
        clf = LocallyLinearEmbedding(n_neighbors=30, n_components=dim, method='standard', random_state=42)
        X_lle = clf.fit_transform(X)
        reconstruction_error[dim] = clf.reconstruction_error_
    reconstruction_error = pd.Series(reconstruction_error)
    plt.figure()    
    plt.plot(reconstruction_error.loc[dims], 'o-')
    plt.title('Reconstruction Error vs. Number of Components: '+title)
    plt.xlabel('number of components')
    plt.ylabel('reconstruction error')
    plt.savefig('./img/'+'LLE_'+title+'.png')
    plt.close()

############################################################################################################
print('Loading MNIST dataset...')
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
print('MNIST dataset loaded.')
run_PCA('Digits', X_digits, y_digits, 'Digits',10, True)
run_ICA('Digits', X_digits, y_digits, 'Digits', 10, [2,5,10,15,20,25,30,35,40,45,50,55,60])
run_RP('Digits', X_digits, y_digits, 'Digits', 10, [2,5,10,15,20,25,30,35,40,45,50,55,60])
run_LLE('Digits', X_digits, y_digits, 'Digits', 10, [2,5,10,15,20,25,30,35,40,45,50,55,60])


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
run_PCA('Cancer', X_cancer, y_cancer, 'Cancer',2, True)
run_ICA('Cancer', X_cancer, y_cancer, 'Cancer', 2, [2,5,10,15,20,25,30])
run_RP('Cancer', X_cancer, y_cancer, 'Cancer', 2, [2,5,10,15,20,25,30])
run_LLE('Cancer', X_cancer, y_cancer, 'Cancer', 2, [2,5,10,15,20,25,30])
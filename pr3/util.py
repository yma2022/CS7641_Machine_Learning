# utility class for the project
from matplotlib import pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import mlrose_hiive as mlrose
import itertools
import time


def plot_curve(data, xlabel="", ylabel=[], title="", show=False):
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle(title)
    ax1.plot(data['k'], data[ylabel[0]], 'o-')
    ax1.set_ylabel(ylabel[0])
    ax2.plot(data['k'], data[ylabel[1]], 'o-')
    ax2.set_ylabel(ylabel[1])
    ax3.plot(data['k'], data[ylabel[2]], 'o-')
    ax3.set_ylabel(ylabel[2])
    ax3.set_xlabel(xlabel)
    if show:
        plt.show()
    else:
        outfile = "img/" + title + ".png"
        plt.savefig(outfile)
        plt.close()


def multiplot_curve(data, method='K-Means', param='SSE', dataset="Digits"):
    plt.figure()
    title = method + " " + param + " vs k " + "Dimensionality reduced " + dataset + " data"
    plt.title(title)
    for f in ['PCA', 'ICA', 'RP', 'LLE']:
        # print(data[f][method]['k'], data[f][method][param])
        plt.plot(data[f][method]['k'], data[f][method][param], 'o-', label=f+'_'+method)
    plt.xlabel('k')
    plt.ylabel(param)
    plt.legend()
    outfile = "img/" + title + ".png"
    plt.savefig(outfile)
    plt.close()

class ImportanceSelect(BaseEstimator, TransformerMixin):
    def __init__(self, model, n=1):
         self.model = model
         self.n = n
    def fit(self, *args, **kwargs):
         self.model.fit(*args, **kwargs)
         return self
    def transform(self, X):
         return X[:,self.model.feature_importances_.argsort()[::-1][:self.n]]
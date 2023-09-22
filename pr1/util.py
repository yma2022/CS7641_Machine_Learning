# utility class for the project
from matplotlib import pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
import numpy as np
from sklearn.model_selection import KFold


def plot_learning_curve(dataset, estimator, X_train, y_train, param_range, title="", ylim=(0.5, 1.01), show=False):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X_train, y_train, cv=4, n_jobs=4, train_sizes=param_range, verbose=0)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.ylim(ylim)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1,
                    color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
            label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
            label="Cross-validation score")

    plt.legend(loc="best")
    if show:
        plt.show()
    else:
        outfile = "img/" + dataset + '_' + title + ".png"
        plt.savefig(outfile)
        plt.close()

def plot_validation_curve(dataset, clf_dt, X_train, y_train, title, xlabel, p_name="max_depth", p_range=np.arange(1, 31), cv=4, log=False, ylim=(0.5, 1.01), show=False):
    plt.figure()
    train_scores, valid_scores = validation_curve(clf_dt, X_train, y_train, param_name=p_name, param_range=p_range, cv=cv, n_jobs=-1, verbose=0)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(valid_scores, axis=1)
    test_scores_std = np.std(valid_scores, axis=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Score")
    plt.grid()
    plt.ylim(ylim)
    plt.fill_between(p_range, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1,
                    color="r")
    plt.fill_between(p_range, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
    if log:
        plt.semilogx(p_range, train_scores_mean, 'o-', label="Training score",
            color="r")
        plt.semilogx(p_range, test_scores_mean, 'o-', label="Cross-validation score",
            color="g")
    else:
        plt.plot(p_range, train_scores_mean, 'o-', label="Training score",
            color="r")
        plt.plot(p_range, test_scores_mean, 'o-', label="Cross-validation score",
            color="g")

    plt.xticks(p_range)
    plt.legend(loc="best")
    if show:
        plt.show()
    else:
        outfile = "img/"+ dataset + '_'  + title + ".png"
        plt.savefig(outfile)
        plt.close()
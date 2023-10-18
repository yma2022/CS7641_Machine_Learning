# utility class for the project
from matplotlib import pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
import numpy as np
import mlrose_hiive as mlrose
import itertools
import time


def plot_curve(data, param_range, ylabel="", title="", show=False):
    plt.figure()
    plt.title(title)
    plt.xlabel("# of Iterations")
    plt.ylabel(ylabel)
    labels = ["Back Prop", "RHC", "SA", "GA"]
    colors = ['r', 'g', 'b', 'y']
    for i in range(len(data)):
        plt.plot(param_range, data[i],'o-', color=colors[i], label=labels[i])
    plt.grid()
    plt.legend(loc="best")
    if show:
        plt.show()
    else:
        outfile = "img/" + title + ".png"
        plt.savefig(outfile)
        plt.close()


def plot_problem(data, param_range, ylabel="", title="", show=False):
    plt.figure()
    plt.title(title)
    plt.xlabel("# of Iterations")
    plt.ylabel(ylabel)
    labels = ["RHC", "SA", "GA", "MIMIC"]
    colors = ['r', 'g', 'b', 'y']
    for i in range(len(data)):
        plt.plot(param_range, data[i],'o-', color=colors[i], label=labels[i])
    plt.grid()
    plt.legend(loc="best")
    if show:
        plt.show()
    else:
        outfile = "img/" + title + ".png"
        plt.savefig(outfile)
        plt.close()
def plot_problem_size(data, param_range, ylabel="", title="", show=False):
    plt.figure()
    plt.title(title)
    plt.xlabel("Problem Size")
    plt.ylabel(ylabel)
    labels = ["RHC", "SA", "GA", "MIMIC"]
    colors = ['r', 'g', 'b', 'y']
    for i in range(len(data)):
        plt.plot(param_range, data[i],'o-', color=colors[i], label=labels[i])
    plt.grid()
    plt.legend(loc="best")
    if show:
        plt.show()
    else:
        outfile = "img/" + title + ".png"
        plt.savefig(outfile)
        plt.close()
        
def rhc_tuning(problem):
    restarts = None
    best_fitness = None
    for i in [0, 25, 50, 75, 100]:
        _, rhc_best_fitness, _ = mlrose.random_hill_climb(problem, max_attempts = 100, max_iters = 1000, curve = True,random_state=42, restarts=i)
        if not restarts:
            restarts = i
            best_fitness = rhc_best_fitness
        elif rhc_best_fitness > best_fitness:
            restarts = i
            best_fitness = rhc_best_fitness
    return restarts

def sa_tuning(problem):
    #init_temp, exp_const, min_temp
    best_param = None
    best_fitness = None
    sa_hyperparams = [[1, 2, 4, 8, 16, 32, 64],[0.1, 0.2, 0.4, 0.8],[0.001, 0.01, 0.1, 1]]

    for i in itertools.product(*sa_hyperparams):
        decay = mlrose.ExpDecay(init_temp = i[0], exp_const=i[1], min_temp=i[2])
        _, sa_best_fitness, _ = mlrose.simulated_annealing(problem, schedule = decay,max_attempts = 100, max_iters = 1000, curve = True,random_state=42)
        if not best_param:
            best_param = i
            best_fitness = sa_best_fitness
        elif sa_best_fitness > best_fitness:
            best_param = i
            best_fitness = sa_best_fitness
    return best_param

def ga_tuning(problem):
    #pop_size, mutation_prob
    best_param = None
    best_fitness = None
    ga_hyperparams = [[50, 100, 200, 400, 800],[0.1, 0.2, 0.4, 0.8]]

    for i in itertools.product(*ga_hyperparams):
        _, ga_best_fitness, _ = mlrose.genetic_alg(problem, pop_size=i[0], mutation_prob=i[1], max_attempts=100, max_iters=1000, curve=True, random_state=42)
        if not best_param:
            best_param = i
            best_fitness = ga_best_fitness
        elif ga_best_fitness > best_fitness:
            best_param = i
            best_fitness = ga_best_fitness
    return best_param

def mimic_tuning(problem):
    #pop_size, keep_pct
    best_param = None
    best_fitness = None
    mimic_hyperparams = [[50, 100, 200, 400, 800],[0.1, 0.2, 0.4, 0.8]]

    for i in itertools.product(*mimic_hyperparams):
        _, mimic_best_fitness, _ = mlrose.mimic(problem, pop_size=i[0], keep_pct=i[1], max_attempts=100, max_iters=1000, curve=True, random_state=42)
        if not best_param:
            best_param = i
            best_fitness = mimic_best_fitness
        elif mimic_best_fitness > best_fitness:
            best_param = i
            best_fitness = mimic_best_fitness
    return best_param
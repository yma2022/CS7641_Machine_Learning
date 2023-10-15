import numpy as np
import mlrose_hiive as mlrose
import time
import util
import random

SEEDs = [42, 24, 7, 100, 200, 300, 400, 500, 600, 700]

weights = [10, 5, 2, 8, 15, 3, 7, 9, 11, 1, 4, 6, 12, 13, 14, 16, 17, 18, 19, 20]
values = [i for i in range(1, 21)]
max_weight_pct = 0.6
fitness = mlrose.Knapsack(weights, values, max_weight_pct)
problem = mlrose.DiscreteOpt(length=len(values), fitness_fn=fitness, maximize=True, max_val=2)

# Select and run a randomized optimization algorithm.


iterations = [5, 25, 50, 75, 100, 125, 150]
data_time = []
data_fitness = []

restarts = util.rhc_tuning(problem)
print("RHC restarts for knapsack: ", restarts)
process_time_rhc = [0] * len(iterations)
best_fitness_rhc = [0] * len(iterations)
for k in range(10):
    for i, iter in enumerate(iterations):
        start = time.time()
        best_state, best_fitness, fitness_curve1 = mlrose.random_hill_climb(problem, max_attempts = 100, max_iters = iter, curve = True,random_state=SEEDs[k], restarts=restarts)
        stop = time.time()
        process_time_rhc[i] += stop - start
        best_fitness_rhc[i] += best_fitness
        if k == 9:
            process_time_rhc[i] /= 10
            best_fitness_rhc[i] /= 10
data_time.append(process_time_rhc)
data_fitness.append(best_fitness_rhc)

decay_param = util.sa_tuning(problem)
print("SA decay param for knapsack: ", decay_param)
# Define decay schedule
schedule = mlrose.ExpDecay(init_temp=decay_param[0], exp_const=decay_param[1], min_temp=decay_param[2])
process_time_sa = [0] * len(iterations)
best_fitness_sa = [0] * len(iterations)
for k in range(10):
    for i, iter in enumerate(iterations):
        start = time.time()
        best_state, best_fitness, fitness_curve2 = mlrose.simulated_annealing(problem, schedule = schedule,max_attempts = 100, max_iters = iter, curve = True,random_state = SEEDs[k])
        stop = time.time()
        process_time_sa[i] += stop - start
        best_fitness_sa[i] += best_fitness
        if k == 9:
            process_time_sa[i] /= 10
            best_fitness_sa[i] /= 10
data_time.append(process_time_sa)
data_fitness.append(best_fitness_sa)


pop_ga, mutation_p = util.ga_tuning(problem)
print("GA pop size for knapsack: ", pop_ga, " and mutation prob: ", mutation_p)
process_time_ga = [0] * len(iterations)
best_fitness_ga = [0] * len(iterations)
for k in range(10):
    for i, iter in enumerate(iterations):
        start = time.time()
        best_state, best_fitness, fitness_curve3 = mlrose.genetic_alg(problem, pop_size=pop_ga, mutation_prob=mutation_p, max_attempts = 100, max_iters = iter, curve = True, random_state = SEEDs[k])
        stop = time.time()
        process_time_ga[i] += stop - start
        best_fitness_ga[i] += best_fitness
        if k == 9:
            process_time_ga[i] /= 10
            best_fitness_ga[i] /= 10
data_time.append(process_time_ga)
data_fitness.append(best_fitness_ga)

pop_m, keep_p = util.mimic_tuning(problem)
print("MIMIC pop size for knapsack: ", pop_m, " and keep pct: ", keep_p)
process_time_mimic = [0] * len(iterations)
best_fitness_mimic = [0] * len(iterations)
for k in range(10):
    for i, iter in enumerate(iterations):
        start = time.time()
        best_state, best_fitness, fitness_curve4 = mlrose.mimic(problem, pop_size=pop_m, keep_pct=keep_p, max_attempts = 100, max_iters = iter, curve = True, random_state = SEEDs[k])
        stop = time.time()
        process_time_mimic[i] += stop - start
        best_fitness_mimic[i] += best_fitness
        if k == 9:
            process_time_mimic[i] /= 10
            best_fitness_mimic[i] /= 10
data_time.append(process_time_mimic)
data_fitness.append(best_fitness_mimic)

print("iterations: ", iterations)
print("train_data: ", data_fitness)
print("process_time: ", data_time)
util.plot_problem(data_fitness, iterations, ylabel="Best Fitness", title="Knapsack Optimal by Iterations: Average of 10", show=False)
util.plot_problem(data_time, iterations, ylabel="Process Time", title="Knapsack Process Time: Average of 10", show=False)



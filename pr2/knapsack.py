import numpy as np
import mlrose_hiive as mlrose
import time
import util
import random

SEEDs = [42, 24, 7, 100, 200, 300, 400, 500, 600, 700]

def items(size):
    weights = [random.randint(1, 50) for _ in range(size)]
    values = [random.randint(1, 20) for _ in range(size)]
    return weights, values
np.random.seed(902764819)
weights, values = items(20)
max_weight_pct = 0.6
fitness = mlrose.Knapsack(weights, values, max_weight_pct)
problem = mlrose.DiscreteOpt(length=len(values), fitness_fn=fitness, maximize=True, max_val=2)

# Select and run a randomized optimization algorithm.


iterations = [5, 25, 50, 75, 100, 125, 150]
# data_time = []
# data_fitness = []
# fevals = []

# restarts = util.rhc_tuning(problem)
# print("RHC restarts for knapsack: ", restarts)
# process_time_rhc = [0] * len(iterations)
# best_fitness_rhc = [0] * len(iterations)
# fevals_rhc = [0] * len(iterations)
# for k in range(10):
#     for i, iter in enumerate(iterations):
#         start = time.time()
#         best_state, best_fitness, fitness_curve1 = mlrose.random_hill_climb(problem, max_attempts = 100, max_iters = iter, curve = True,random_state=SEEDs[k], restarts=restarts)
#         stop = time.time()
#         process_time_rhc[i] += stop - start
#         best_fitness_rhc[i] += best_fitness
#         fevals_rhc[i] += len(fitness_curve1)
#         if k == 9:
#             process_time_rhc[i] /= 10
#             best_fitness_rhc[i] /= 10
#             fevals_rhc[i] /= 10
# data_time.append(process_time_rhc)
# data_fitness.append(best_fitness_rhc)
# fevals.append(fevals_rhc)

# decay_param = util.sa_tuning(problem)
# print("SA decay param for knapsack: ", decay_param)
# # Define decay schedule
# schedule = mlrose.ExpDecay(init_temp=decay_param[0], exp_const=decay_param[1], min_temp=decay_param[2])
# process_time_sa = [0] * len(iterations)
# best_fitness_sa = [0] * len(iterations)
# fevals_sa = [0] * len(iterations)
# for k in range(10):
#     for i, iter in enumerate(iterations):
#         start = time.time()
#         best_state, best_fitness, fitness_curve2 = mlrose.simulated_annealing(problem, schedule = schedule,max_attempts = 100, max_iters = iter, curve = True,random_state = SEEDs[k])
#         stop = time.time()
#         process_time_sa[i] += stop - start
#         best_fitness_sa[i] += best_fitness
#         fevals_sa[i] += len(fitness_curve2)
#         if k == 9:
#             process_time_sa[i] /= 10
#             best_fitness_sa[i] /= 10
#             fevals_sa[i] /= 10
# data_time.append(process_time_sa)
# data_fitness.append(best_fitness_sa)
# fevals.append(fevals_sa)


# pop_ga, mutation_p = util.ga_tuning(problem)
# print("GA pop size for knapsack: ", pop_ga, " and mutation prob: ", mutation_p)
# process_time_ga = [0] * len(iterations)
# best_fitness_ga = [0] * len(iterations)
# fevals_ga = [0] * len(iterations)
# for k in range(10):
#     for i, iter in enumerate(iterations):
#         start = time.time()
#         best_state, best_fitness, fitness_curve3 = mlrose.genetic_alg(problem, pop_size=pop_ga, mutation_prob=mutation_p, max_attempts = 100, max_iters = iter, curve = True, random_state = SEEDs[k])
#         stop = time.time()
#         process_time_ga[i] += stop - start
#         best_fitness_ga[i] += best_fitness
#         fevals_ga[i] += len(fitness_curve3)
#         if k == 9:
#             process_time_ga[i] /= 10
#             best_fitness_ga[i] /= 10
#             fevals_ga[i] /= 10
# data_time.append(process_time_ga)
# data_fitness.append(best_fitness_ga)
# fevals.append(fevals_ga)

# pop_m, keep_p = util.mimic_tuning(problem)
# print("MIMIC pop size for knapsack: ", pop_m, " and keep pct: ", keep_p)
# process_time_mimic = [0] * len(iterations)
# best_fitness_mimic = [0] * len(iterations)
# fevals_mimic = [0] * len(iterations)
# for k in range(10):
#     for i, iter in enumerate(iterations):
#         start = time.time()
#         best_state, best_fitness, fitness_curve4 = mlrose.mimic(problem, pop_size=pop_m, keep_pct=keep_p, max_attempts = 100, max_iters = iter, curve = True, random_state = SEEDs[k])
#         stop = time.time()
#         process_time_mimic[i] += stop - start
#         best_fitness_mimic[i] += best_fitness
#         fevals_mimic[i] += len(fitness_curve4)
#         if k == 9:
#             process_time_mimic[i] /= 10
#             best_fitness_mimic[i] /= 10
#             fevals_mimic[i] /= 10
# data_time.append(process_time_mimic)
# data_fitness.append(best_fitness_mimic)
# fevals.append(fevals_mimic)

# print("iterations: ", iterations)
# print("train_data: ", data_fitness)
# print("process_time: ", data_time)
# print("fevals: ", fevals)
# util.plot_problem(data_fitness, iterations, ylabel="Best Fitness", title="Knapsack Optimal by Iterations: Average of 10", show=False)
# util.plot_problem(data_time, iterations, ylabel="Process Time", title="Knapsack Process Time: Average of 10", show=False)
# util.plot_problem(fevals, iterations, ylabel="# of Evaluations", title="Knapsack Function Evaluations: Average of 10", show=False)


problem_sizes = [10,40, 70, 100, 130]
data_size = []

size_fitness_rhc = [0]*5
size_fitness_sa = [0]*5
size_fitness_ga = [0]*5
size_fitness_mimic = [0]*5

for i, s in enumerate(problem_sizes):
    weights, values = items(problem_sizes[i])
    max_weight_pct = 0.6
    fitness = mlrose.Knapsack(weights, values, max_weight_pct)
    problem = mlrose.DiscreteOpt(length=len(values), fitness_fn=fitness, maximize=True, max_val=2)
    restarts = util.rhc_tuning(problem)
    print("RHC restarts for knapsack: ", restarts)

    decay_param = util.sa_tuning(problem)
    print("SA decay param for knapsack: ", decay_param)
    # Define decay schedule
    schedule = mlrose.ExpDecay(init_temp=decay_param[0], exp_const=decay_param[1], min_temp=decay_param[2])

    pop_ga, mutation_p = util.ga_tuning(problem)
    print("GA pop size and mutation prob for knapsack: ", pop_ga, mutation_p)

    pop_m, keep_p = util.mimic_tuning(problem)
    print("MIMIC pop size and keep pct for knapsack: ", pop_m, keep_p)
    for k in range(10):
        best_state1, best_fitness1, fitness_curve1 = mlrose.random_hill_climb(problem, max_attempts = 100, max_iters = 200, curve = True, random_state=SEEDs[k], restarts=restarts)
        size_fitness_rhc[i] += best_fitness1
        if k == 9:
            size_fitness_rhc[i] /= 10

        best_state2, best_fitness2, fitness_curve2 = mlrose.simulated_annealing(problem, schedule = schedule,max_attempts = 100, max_iters = 200, curve = True,random_state = SEEDs[k])
        size_fitness_sa[i] += best_fitness2
        if k == 9:
            size_fitness_sa[i] /= 10

        best_state3, best_fitness3, fitness_curve3 = mlrose.genetic_alg(problem, pop_size=pop_ga, mutation_prob=mutation_p, max_attempts = 100, max_iters = 200, curve = True, random_state = SEEDs[k])
        size_fitness_ga[i] += best_fitness3
        if k == 9:
            size_fitness_ga[i] /= 10

        best_state4, best_fitness4, fitness_curve4 = mlrose.mimic(problem, pop_size=pop_m, keep_pct=keep_p, max_attempts = 100, max_iters = 200, curve = True, random_state = SEEDs[k])
        size_fitness_mimic[i] += best_fitness4
        if k == 9:
            size_fitness_mimic[i] /= 10

data_size.append(size_fitness_rhc)
data_size.append(size_fitness_sa)
data_size.append(size_fitness_ga)
data_size.append(size_fitness_mimic)
print("data_size: ", data_size)
print("start plotting size vs fitness")
util.plot_problem_size(data_size, [10, 20, 30, 40, 50], ylabel="Best Fitness", title="Knapsack Optimal by Size: Average of 10", show=False)
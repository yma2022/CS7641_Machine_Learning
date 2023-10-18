import numpy as np
import mlrose_hiive as mlrose
import time
import util

SEEDs = [42, 24, 7, 100, 200, 300, 400, 500, 600, 700]

# Create list of city coordinates
coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]

# Initialize fitness function object using coords_list
fitness_coords = mlrose.TravellingSales(coords = coords_list)

# Create list of distances between pairs of cities
dist_list = [(0, 1, 3.1623), (0, 2, 4.1231), (0, 3, 5.8310), (0, 4, 4.2426), \
             (0, 5, 5.3852), (0, 6, 4.0000), (0, 7, 2.2361), (1, 2, 1.0000), \
             (1, 3, 2.8284), (1, 4, 2.0000), (1, 5, 4.1231), (1, 6, 4.2426), \
             (1, 7, 2.2361), (2, 3, 2.2361), (2, 4, 2.2361), (2, 5, 4.4721), \
             (2, 6, 5.0000), (2, 7, 3.1623), (3, 4, 2.0000), (3, 5, 3.6056), \
             (3, 6, 5.0990), (3, 7, 4.1231), (4, 5, 2.2361), (4, 6, 3.1623), \
             (4, 7, 2.2361), (5, 6, 2.2361), (5, 7, 3.1623), (6, 7, 2.2361)]

# Initialize fitness function object using dist_list
fitness_dists = mlrose.TravellingSales(distances = dist_list)

# Define optimization problem object
problem = mlrose.TSPOpt(length = 8, fitness_fn = fitness_coords, maximize=False)


# Select and run a randomized optimization algorithm.




iterations = [5, 50, 100, 150, 200, 250]
data_time = []
data_fitness = []
fevals = []


restarts = util.rhc_tuning(problem)
print("RHC restarts for 8 queens: ", restarts)
process_time_rhc = [0] * len(iterations)
best_fitness_rhc = [0] * len(iterations)
fevals_rhc = [0] * len(iterations)
for k in range(10):
    for i, iter in enumerate(iterations):
        start = time.time()
        best_state, best_fitness, fitness_curve1 = mlrose.random_hill_climb(problem, max_attempts = 100, max_iters = iter, curve = True,random_state=SEEDs[k], restarts=restarts)
        stop = time.time()
        process_time_rhc[i] += stop - start
        best_fitness_rhc[i] += best_fitness
        fevals_rhc[i] += len(fitness_curve1)
        if k == 9:
            process_time_rhc[i] /= 10
            best_fitness_rhc[i] /= 10
            fevals_rhc[i] /= 10
       
data_time.append(process_time_rhc)
data_fitness.append(best_fitness_rhc)
fevals.append(fevals_rhc)


decay_param = util.sa_tuning(problem)
print("SA decay param for 8 queens: ", decay_param)
# Define decay schedule
schedule = mlrose.ExpDecay(init_temp=decay_param[0], exp_const=decay_param[1], min_temp=decay_param[2])
process_time_sa = [0] * len(iterations)
best_fitness_sa = [0] * len(iterations)
fevals_sa = [0] * len(iterations)
for k in range(10):
    for i, iter in enumerate(iterations):
        start = time.time()
        best_state, best_fitness, fitness_curve2 = mlrose.simulated_annealing(problem, schedule = schedule,max_attempts = 100, max_iters = iter, curve = True,random_state = SEEDs[k])
        stop = time.time()
        process_time_sa[i] += stop - start
        best_fitness_sa[i] += best_fitness
        fevals_sa[i] += len(fitness_curve2)
        if k == 9:
            process_time_sa[i] /= 10
            best_fitness_sa[i] /= 10
            fevals_sa[i] /= 10

data_time.append(process_time_sa)
data_fitness.append(best_fitness_sa)
fevals.append(fevals_sa)


pop_ga, mutation_p = util.ga_tuning(problem)
print("GA pop size and mutation prob for 8 queens: ", pop_ga, mutation_p)
process_time_ga = [0] * len(iterations)
best_fitness_ga = [0] * len(iterations)
fevals_ga = [0] * len(iterations)
for k in range(10):
    for i, iter in enumerate(iterations):
        start = time.time()
        best_state, best_fitness, fitness_curve3 = mlrose.genetic_alg(problem, pop_size=pop_ga, mutation_prob=mutation_p, max_attempts = 100, max_iters = iter, curve = True, random_state = SEEDs[k])
        stop = time.time()
        process_time_ga[i] += stop - start
        best_fitness_ga[i] += best_fitness
        fevals_ga[i] += len(fitness_curve3)
        if k == 9:
            process_time_ga[i] /= 10
            best_fitness_ga[i] /= 10
            fevals_ga[i] /= 10
data_time.append(process_time_ga)
data_fitness.append(best_fitness_ga)
fevals.append(fevals_ga)

pop_m, keep_p = util.mimic_tuning(problem)
print("MIMIC pop size and keep pct for 8 queens: ", pop_m, keep_p)
process_time_mimic = [0] * len(iterations)
best_fitness_mimic = [0] * len(iterations)
fevals_mimic = [0] * len(iterations)
for k in range(10):
    for i, iter in enumerate(iterations):
        start = time.time()
        best_state, best_fitness, fitness_curve4 = mlrose.mimic(problem, pop_size=pop_m, keep_pct=keep_p, max_attempts = 100, max_iters = iter, curve = True, random_state = SEEDs[k])
        stop = time.time()
        process_time_mimic[i] += stop - start
        best_fitness_mimic[i] += best_fitness
        fevals_mimic[i] += len(fitness_curve4)
        if k == 9:
            process_time_mimic[i] /= 10
            best_fitness_mimic[i] /= 10
            fevals_mimic[i] /= 10
data_time.append(process_time_mimic)
data_fitness.append(best_fitness_mimic)
fevals.append(fevals_mimic)

print("iterations: ", iterations)
print("train_data: ", data_fitness)
print("process_time: ", data_time)

util.plot_problem(data_fitness, iterations, ylabel="Best Fitness", title="TSP Optimal by Iterations: Average of 10", show=False)
util.plot_problem(data_time, iterations, ylabel="Process Time", title="TSP Process Time: Average of 10", show=False)
util.plot_problem(fevals, iterations, ylabel="Function Evaluations", title="TSP Function Evaluations: Average of 10", show=False)



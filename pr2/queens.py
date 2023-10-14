import numpy as np
import mlrose_hiive as mlrose
import time
import util

SEEDs = [42, 24, 7, 100, 200, 300, 400, 500, 600, 700]

# Define a fitness function object
def queens_max(state):

   # Initialize counter
    fitness_cnt = 0

    # For all pairs of queens
    for i in range(len(state) - 1):
        for j in range(i + 1, len(state)):
        # Check for horizontal, diagonal-up and diagonal-down attacks
            if (state[j] != state[i]) and (state[j] != state[i] + (j - i)) and (state[j] != state[i] - (j - i)):
                # If no attacks, then increment counter
                fitness_cnt += 1

    return fitness_cnt


# Initialize custom fitness function object
fitness_cust = mlrose.CustomFitness(queens_max)


# Define an optimization problem object.
# The 8-Queens problem is an example of a discrete-state optimization problem, 
# since each of the elements of the state vector must take on an integer value in the range 0 to 7.

problem = mlrose.DiscreteOpt(length = 8, fitness_fn = fitness_cust, maximize = True, max_val = 8)

# Select and run a randomized optimization algorithm.




iterations = [5, 50, 100, 150, 200, 250]
data_time = []
data_fitness = []


restarts = util.rhc_tuning(problem)
print("RHC restarts for 8 queens: ", restarts)
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
print("SA decay param for 8 queens: ", decay_param)
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

util.plot_problem(data_fitness, iterations, ylabel="Best Fitness", title="8 Queens Optimal by Iterations: Average of 10", show=False)
util.plot_problem(data_time, iterations, ylabel="Process Time", title="8 Queens Process Time: Average of 10", show=False)



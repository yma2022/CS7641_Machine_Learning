import numpy as np
import gym
import time
import matplotlib.pyplot as plt
from gym.envs.toy_text.frozen_lake import generate_random_map
from tqdm import tqdm
from util import plot_blackjack_policy, decay_schedule, plot_policy_map
from planner import extract_policy, policy_iteration, value_iteration
import os
import warnings

import gym
import pygame
import pickle

class Blackjack:
    def __init__(self):
        self._env = gym.make('Blackjack-v1', render_mode=None)
        # Explanation of convert_state_obs lambda:
        # def function(state, done):
        # 	if done:
		#         return -1
        #     else:
        #         if state[2]:
        #             int(f"{state[0]+6}{(state[1]-2)%10}")
        #         else:
        #             int(f"{state[0]-4}{(state[1]-2)%10}")
        self._convert_state_obs = lambda state, done: (
            -1 if done else int(f"{state[0] + 6}{(state[1] - 2) % 10}") if state[2] else int(
                f"{state[0] - 4}{(state[1] - 2) % 10}"))
        # Transitions and rewards matrix from: https://github.com/rhalbersma/gym-blackjack-v1
        current_dir = os.path.dirname('/home/johnd/CS7641/pr4/')
        file_name = 'blackjack-envP'
        f = os.path.join(current_dir, file_name)
        try:
            self._P = pickle.load(open(f, "rb"))
        except IOError:
            print("Pickle load failed.  Check path", f)
        self._n_actions = self.env.action_space.n
        self._n_states = len(self._P)

    @property
    def n_actions(self):
        return self._n_actions

    @n_actions.setter
    def n_actions(self, n_actions):
        self._n_actions = n_actions

    @property
    def n_states(self):
        return self._n_states

    @n_states.setter
    def n_states(self, n_states):
        self._n_states = n_states

    @property
    def P(self):
        return self._P

    @P.setter
    def P(self, P):
        self._P = P

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, env):
        self._env = env

    @property
    def convert_state_obs(self):
        return self._convert_state_obs

    @convert_state_obs.setter
    def convert_state_obs(self, convert_state_obs):
        self._convert_state_obs = convert_state_obs

def Blackjack_Experiments():
    blackjack = Blackjack()
    print(blackjack.env.reset())
    time_array = [0] * 10
    gamma_arr = [0] * 10
    iters = [0] * 10
    list_scores = [0] * 10
    gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    ### POLICY ITERATION ####
    print('POLICY ITERATION WITH BLACKJACK')
    for i, g in enumerate(gammas):
        st = time.time()
        best_policy, k = policy_iteration(blackjack, gamma=g)
        end = time.time()
        print('policy iteration finish for gamma:', g)
        scores = evaluate_policy(blackjack, best_policy, gamma=g)
        # print(best_policy)
        plot_blackjack_policy(blackjack, best_policy.reshape(29,10), 'Blackjack Policy Map Iteration ' + str(i) + ' (Policy Iteration) ' + 'Gamma ' + str(g), directions_blackjack())
        print('policy evaluation finish for gamma:', g)
        gamma_arr[i] = g
        list_scores[i] = np.mean(scores)
        iters[i] = k
        time_array[i] = end - st


    plt.plot(gamma_arr, time_array)
    plt.xlabel('Gamma')
    plt.title('Blackjack Policy Iteration Execution Time Analysis')
    plt.ylabel('Execution Time (s)')
    plt.grid()
    plt.savefig('img/Blackjack Policy Iteration Execution Time Analysis.png')
    plt.close()

    plt.plot(gamma_arr, list_scores)
    plt.xlabel('Gamma')
    plt.ylabel('Average Rewards')
    plt.title('Blackjack Policy Iteration Reward Analysis')
    plt.grid()
    plt.savefig('img/Blackjack Policy Iteration Reward Analysis.png')
    plt.close()

    plt.plot(gamma_arr, iters)
    plt.xlabel('Gamma')
    plt.ylabel('Iterations to Converge')
    plt.ylim(0,10)
    plt.title('Blackjack - Policy Iteration - Convergence Analysis')
    plt.grid()
    plt.savefig('img/Blackjack Policy Iteration Convergence Analysis.png')
    plt.close()

    ### VALUE ITERATION ###
    print('VALUE ITERATION WITH FROZEN LAKE')
    for i, g in enumerate(gammas):
        st = time.time()
        best_value, k = value_iteration(blackjack, gamma=g)
        end = time.time()
        print('value iteration finish for gamma: ', g)
        policy = extract_policy(blackjack, best_value, gamma=g)
        plot_blackjack_policy(blackjack, policy.reshape(29,10), 'Blackjack Policy Map Iteration ' + str(i) + ' (Value Iteration) ' + 'Gamma ' + str(g), directions_blackjack())
        policy_score = evaluate_policy(blackjack, policy, gamma=g)
        print('policy evaluation finish for gamma: ', g)
        gamma_arr[i] = g
        iters[i] = k
        list_scores[i] = np.mean(policy_score)
        time_array[i] = end - st

    plt.plot(gamma_arr, time_array)
    plt.xlabel('Gammas')
    plt.title('Blackjack Value Iteration Execution Time Analysis')
    plt.ylabel('Execution Time (s)')
    plt.grid()
    plt.savefig('img/Blackjack Value Iteration Execution Time Analysis.png')
    plt.close()

    plt.plot(gamma_arr, list_scores)
    plt.xlabel('Gammas')
    plt.ylabel('Average Rewards')
    plt.title('Blackjack Value Iteration Reward Analysis')
    plt.grid()
    plt.savefig('img/Blackjack Value Iteration Reward Analysis.png')
    plt.close()

    plt.plot(gamma_arr, iters)
    plt.xlabel('Gammas')
    plt.ylabel('Iterations to Converge')
    plt.title('Blackjack Value Iteration Convergence Analysis')
    plt.grid()
    plt.savefig('img/Blackjack Value Iteration Convergence Analysis.png')
    plt.close()

    ### Q-LEARNING #####
    print('Q LEARNING WITH FROZEN LAKE')
    st = time.time()
    reward_array = []
    iter_array = []
    size_array = []
    rewards_chunks_array = []
    rewards_averages_array = []
    iters_chunks_array = []
    iters_averages_array = []
    time_array = []
    Q_array = []
    for epsilon in [0.1, 0.3, 0.5, 0.7, 0.9]:
        Q = np.zeros((blackjack.n_states, blackjack.n_actions))
        rewards = []
        iters = []
        optimal = [0] * blackjack.n_states
        alpha = 0.1
        gamma = 0.9
        episodes = 30000
        epsilons = decay_schedule(epsilon, 0.1, 0.9, episodes)
        alphas = decay_schedule(alpha, 0.01, 0.9, episodes)
        for episode in tqdm(range(episodes), leave=False):
            state, info = blackjack.env.reset()
            done = False
            state = blackjack.convert_state_obs(state, done)
            t_reward = 0
            step_idx = 0
            while not done:
                step_idx += 1       
                if np.random.rand() > (epsilons[episode]):
                    action = np.argmax(Q[state])
                else:
                    action = np.random.randint(len(Q[state]))
                
                next_state, reward, terminated, truncated, _ = blackjack.env.step(action)                
                done = terminated or truncated
                next_state = blackjack.convert_state_obs(next_state, done)
                td_target = reward + gamma * np.max(Q[next_state])* (not done)
                td_delta = td_target - Q[state, action]
                Q[state, action] += alphas[episode] * td_delta
                t_reward += reward
                state = next_state
                # print(step_idx)
            rewards.append(t_reward)
            iters.append(step_idx)
            # print(episode)
        print(epsilon)
        end = time.time()
        print("time :", end - st)
        time_array.append(end - st)

        for state, action in enumerate(np.argmax(Q, axis=1)):
            optimal[state] = action
        # print(optimal)
        # print(optimal)
        plot_blackjack_policy(blackjack, np.array(optimal).reshape(29,10), 'Blackjack Policy Map Iteration '+ str(epsilon) + ' (Q Learning) ' + 'Epsilon '+ str(epsilon), directions_blackjack())
        reward_array.append(rewards)
        iter_array.append(iters)
        Q_array.append(Q)

        # Plot results
        def chunk_list(l, n):
            for i in range(0, len(l), n):
                yield l[i:i + n]

        size = int(episodes / 50)
        size_array.append(size)

        chunks = list(chunk_list(rewards, size))
        averages = [sum(chunk) / len(chunk) for chunk in chunks]        
        rewards_chunks_array.append(chunks)
        rewards_averages_array.append(averages)

        chunks = list(chunk_list(iters, size))
        averages = [sum(chunk) / len(chunk) for chunk in chunks]
        iters_chunks_array.append(chunks)
        iters_averages_array.append(averages)

    plt.plot(range(0, len(reward_array[0]), size_array[0]), rewards_averages_array[0], label='epsilon=0.1')
    plt.plot(range(0, len(reward_array[1]), size_array[1]), rewards_averages_array[1], label='epsilon=0.3')
    plt.plot(range(0, len(reward_array[2]), size_array[2]), rewards_averages_array[2], label='epsilon=0.5')
    plt.plot(range(0, len(reward_array[3]), size_array[3]), rewards_averages_array[3], label='epsilon=0.7')
    plt.plot(range(0, len(reward_array[4]), size_array[4]), rewards_averages_array[4], label='epsilon=0.9')
    plt.legend()

    plt.xlabel('# of Episodes')
    plt.grid()
    plt.title('Blackjack Q Learning Rewards Analysis')
    plt.ylabel('Average Reward')
    plt.savefig('img/Blackjack Q Learning Rewards Analysis.png')
    plt.close()

    plt.plot(range(0, len(iter_array[0]), size_array[0]), iters_averages_array[0], label='epsilon=0.1')
    plt.plot(range(0, len(iter_array[1]), size_array[1]), iters_averages_array[1], label='epsilon=0.3')
    plt.plot(range(0, len(iter_array[2]), size_array[2]), iters_averages_array[2], label='epsilon=0.5')
    plt.plot(range(0, len(iter_array[3]), size_array[3]), iters_averages_array[3], label='epsilon=0.7')
    plt.plot(range(0, len(iter_array[4]), size_array[4]), iters_averages_array[4], label='epsilon=0.9')
    plt.legend()
    plt.xlabel('# of Episodes')
    plt.grid()
    plt.title('Blackjack Q Learning Iterations Analysis')
    plt.ylabel('Average Iterations')
    plt.savefig('img/Blackjack Q Learning Iterations Analysis.png')
    plt.close()

def run_episode(blackjack, policy, gamma, render=False, desc=None):
    if render:
        env_name = blackjack.env.unwrapped.spec.id
        env = gym.make(env_name, render_mode='human')
    else:
        env = blackjack.env
    total_reward = 0
    done = False
    state, _ = env.reset()
    state = blackjack.convert_state_obs(state, done)
    step_idx = 0
    max_steps = 10000

    while not done and step_idx < max_steps:
        action = int(policy[state])
        # Perform an action and observe how environment acted in response
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = blackjack.convert_state_obs(next_state, done)
        
        # Summarize total reward
        # print(reward)
        total_reward += (gamma ** step_idx * reward)
        # Update current state
        state = next_state
        step_idx += 1
    return total_reward

def colors_lake():
	return {
		b'S': 'green',
		b'F': 'skyblue',
		b'H': 'black',
		b'G': 'gold',
	}

def directions_blackjack():
	return {
		1: 'H',
		0: 'S'
	}


def evaluate_policy(blackjack, policy, gamma, n=1000, desc=None):
    scores = [run_episode(blackjack, policy, gamma, False, desc) for _ in range(n)]
    return np.mean(scores)

print('Starting Blackjack Experiments')
Blackjack_Experiments()
print('Finished Blackjack Experiments')
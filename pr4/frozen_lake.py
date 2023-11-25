import numpy as np
import gym
import time
import matplotlib.pyplot as plt
from gym.envs.toy_text.frozen_lake import generate_random_map
from tqdm import tqdm
from util import plot_policy_map, decay_schedule
from planner import evaluate_policy, extract_policy, policy_iteration, value_iteration


def Frozen_Lake_Experiments():
	# 0 = left; 1 = down; 2 = right;  3 = up
	np.random.seed(42)
	n = 20
	custom_frozen_lake = generate_random_map(size=n)
	frozen_lake = gym.make("FrozenLake-v1", is_slippery=False, desc= custom_frozen_lake)
	print(frozen_lake.observation_space)
	# environment  = 'FrozenLake-v1'
	env = frozen_lake
	print(env.unwrapped.spec.id)
	print(env.observation_space.n, env.action_space.n, len(env.P), len(env.P[0]))
	env = env.unwrapped
	desc = env.unwrapped.desc

	time_array=[0]*10
	gamma_arr=[0]*10
	iters=[0]*10
	list_scores=[0]*10
	gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
	
	### POLICY ITERATION ####
	print('POLICY ITERATION WITH FROZEN LAKE')
	for i, g in enumerate(gammas):
		st=time.time()
		best_policy,k = policy_iteration(env, gamma = g)
		end=time.time()
		print('policy iteration finish for gamma: ', g)
		plot_policy_map('Frozen Lake Policy Map Iteration '+ str(i) + ' (Policy Iteration) ' + 'Gamma '+ str(g),best_policy.reshape(n,n),desc,colors_lake(),directions_lake())
		scores = evaluate_policy(env, best_policy, gamma = g, desc=desc, custom=custom_frozen_lake)
		print('policy evaluation finish for gamma: ', g)		
		gamma_arr[i]= g
		list_scores[i]=np.mean(scores)
		iters[i] = k
		time_array[i]=end-st
	
	
	plt.plot(gamma_arr, time_array)
	plt.xlabel('Gamma')
	plt.title('Frozen Lake Policy Iteration Execution Time Analysis')
	plt.ylabel('Execution Time (s)')
	plt.grid()
	plt.savefig('img/Frozen Lake Policy Iteration Execution Time Analysis.png')
	plt.close()

	plt.plot(gamma_arr,list_scores)
	plt.xlabel('Gamma')
	plt.ylabel('Average Rewards')
	plt.title('Frozen Lake Policy Iteration Reward Analysis')
	plt.grid()
	plt.savefig('img/Frozen Lake Policy Iteration Reward Analysis.png')
	plt.close()

	plt.plot(gamma_arr,iters)
	plt.xlabel('Gamma')
	plt.ylabel('Iterations to Converge')
	plt.title('Frozen Lake Policy Iteration Convergence Analysis')
	plt.grid()
	plt.savefig('img/Frozen Lake Policy Iteration Convergence Analysis.png')
	plt.close()

	
	### VALUE ITERATION ###
	print('VALUE ITERATION WITH FROZEN LAKE')
	for i, g in enumerate(gammas):
		st=time.time()
		best_value,k = value_iteration(env, gamma = g)
		end=time.time()
		print('value iteration finish for gamma: ', g)
		policy = extract_policy(env,best_value, gamma = g)
		policy_score = evaluate_policy(env, policy, gamma=g, desc=desc, custom=custom_frozen_lake)
		print('policy evaluation finish for gamma: ', g)
		plot_policy_map('Frozen Lake Policy Map Iteration '+ str(i) + ' (Value Iteration) ' + 'Gamma '+ str(g),policy.reshape(n,n),desc,colors_lake(),directions_lake())		
		gamma_arr[i]=g
		iters[i]=k
		list_scores[i]=np.mean(policy_score)
		time_array[i]=end-st

	plt.plot(gamma_arr, time_array)
	plt.xlabel('Gamma')
	plt.title('Frozen Lake Value Iteration Execution Time Analysis')
	plt.ylabel('Execution Time (s)')
	plt.grid()
	plt.savefig('img/Frozen Lake Value Iteration Execution Time Analysis.png')
	plt.close()

	plt.plot(gamma_arr,list_scores)
	plt.xlabel('Gamma')
	plt.ylabel('Average Rewards')
	plt.title('Frozen Lake Value Iteration Reward Analysis')
	plt.grid()
	plt.savefig('img/Frozen Lake Value Iteration Reward Analysis.png')
	plt.close()

	plt.plot(gamma_arr,iters)
	plt.xlabel('Gamma')
	plt.ylabel('Iterations to Converge')
	plt.title('Frozen Lake Value Iteration Convergence Analysis')
	plt.grid()
	plt.savefig('img/Frozen Lake Value Iteration Convergence Analysis.png')
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
		Q = np.zeros((env.observation_space.n, env.action_space.n))
		rewards = []
		iters = []
		optimal=[0]*env.observation_space.n
		alpha = 0.1
		gamma = 0.9
		episodes = 30000
		epsilons = decay_schedule(epsilon, 0.1, 0.9, episodes)
		alphas = decay_schedule(alpha, 0.01, 0.9, episodes)
		for episode in tqdm(range(episodes), leave=False):
			state, info = env.reset()
			done = False
			t_reward = 0
			step_idx = 0
			while not done:
				step_idx += 1       
				if np.random.rand() > (epsilons[episode]):
					action = np.argmax(Q[state])
				else:
					action = np.random.randint(len(Q[state]))
				
				next_state, reward, terminated, truncated, _ = env.step(action)
				state_char = desc.flatten()[state]
				next_state_char = desc.flatten()[next_state]
				if next_state_char == b'H':
					reward = -10
				elif next_state_char == b'G':
					reward = 20
					#print('state: ', state_char, ' action: ', action, ' next_state: ', next_state_char, ' reward: ', reward)
				else:
					reward = -0.01
				
				done = terminated or truncated
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
		end=time.time()
		print("time :",end-st)
		time_array.append(end-st)


		for state, action in enumerate(np.argmax(Q, axis=1)):
			optimal[state] = action
		# print(optimal)
		plot_policy_map('Frozen Lake Policy Map Iteration '+ str(epsilon) + ' (Q Learning) ' + 'Epsilon '+ str(epsilon),np.array(optimal).reshape(n,n),desc,colors_lake(),directions_lake())

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

		

	plt.plot(range(0, len(reward_array[0]), size_array[0]), rewards_averages_array[0],label='epsilon=0.1')
	plt.plot(range(0, len(reward_array[1]), size_array[1]), rewards_averages_array[1],label='epsilon=0.3')
	plt.plot(range(0, len(reward_array[2]), size_array[2]), rewards_averages_array[2],label='epsilon=0.5')
	plt.plot(range(0, len(reward_array[3]), size_array[3]), rewards_averages_array[3],label='epsilon=0.7')
	plt.plot(range(0, len(reward_array[4]), size_array[4]), rewards_averages_array[4],label='epsilon=0.9')
	plt.legend()
	plt.xlabel('# of Episodes')
	plt.grid()
	plt.title('Frozen Lake Q Learning Rewards Analysis')
	plt.ylabel('Average Reward')
	plt.savefig('img/Frozen Lake Q Learning Rewards Analysis.png')
	plt.close()


	plt.plot(range(0, len(iter_array[0]), size_array[0]), iters_averages_array[0],label='epsilon=0.1')
	plt.plot(range(0, len(iter_array[1]), size_array[1]), iters_averages_array[1],label='epsilon=0.3')
	plt.plot(range(0, len(iter_array[2]), size_array[2]), iters_averages_array[2],label='epsilon=0.5')
	plt.plot(range(0, len(iter_array[3]), size_array[3]), iters_averages_array[3],label='epsilon=0.7')
	plt.plot(range(0, len(iter_array[4]), size_array[4]), iters_averages_array[4],label='epsilon=0.9')
	plt.legend()
	plt.xlabel('# of Episodes')
	plt.grid()
	plt.title('Frozen Lake Q Learning Iterations Analysis')
	plt.ylabel('Average Iterations')
	plt.savefig('img/Frozen Lake Q Learning Iterations Analysis.png')
	plt.close()
			
def colors_lake():
	return {
		b'S': 'green',
		b'F': 'skyblue',
		b'H': 'black',
		b'G': 'gold',
	}

def directions_lake():
	return {
		3: '⬆',
		2: '➡',
		1: '⬇',
		0: '⬅'
	}

print('STARTING EXPERIMENTS')
Frozen_Lake_Experiments()
print('END OF EXPERIMENTS')
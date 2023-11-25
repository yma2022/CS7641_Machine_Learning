import numpy as np
import gym


def run_episode(env, policy, gamma, render=False, desc=None, custom=None):
	if render:
		env_name = env.unwrapped.spec.id
		env = gym.make(env_name, render_mode='human', desc=custom, is_slippery=False)
	total_reward = 0
	done = False
	state, _ = env.reset()
	step_idx = 0
	max_steps = 10000
	while not done and step_idx < max_steps:
		action = int(policy[state])
		# Perform an action and observe how environment acted in response
		next_state, reward, terminated, truncated, _ = env.step(action)
		done = terminated or truncated
		next_state_char = desc.flatten()[next_state]
		if next_state_char == b'H':
			reward = -10
		elif next_state_char == b'G':
			reward = 20
			#print('state: ', state_char, ' action: ', action, ' next_state: ', next_state_char, ' reward: ', reward)
		else:
			reward = -0.01
		# Summarize total reward
		total_reward += (gamma ** step_idx * reward)
		# Update current state
		state = next_state
		step_idx += 1

	return total_reward


def evaluate_policy(env, policy, gamma, n=100, desc=None, custom=None):
    scores = [run_episode(env, policy, gamma, False, desc, custom) for _ in range(n)]
    return np.mean(scores)

def extract_policy(env, v, gamma):
    policy = np.zeros(len(env.P), dtype=np.float64)
    for state in range(len(env.P)):
        # initialize the Q table for a state
        Q_table = np.zeros(len(env.P[0]), dtype=np.float64)

        # compute Q value for all actions in the state
        for action in range(len(env.P[0])):
            for next_sr in env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                Q_table[action] += (trans_prob * (reward_prob + gamma * v[next_state]))

        # select the action which has maximum Q value as an optimal action of the state
        policy[state] = np.argmax(Q_table)

    return policy

def compute_policy_v(env, policy, gamma):
	v = np.zeros(len(env.P), dtype=np.float64)
	eps = 1e-5
	while True:
		prev_v = np.copy(v)
		for s in range(len(env.P)):
			policy_a = policy[s]
			v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, is_done in env.P[s][policy_a]])
		if (np.sum((np.fabs(prev_v - v))) <= eps):
			break
	return v

def policy_iteration(env, gamma):
	policy = np.random.choice(len(env.P[0]), len(env.P))  
	max_iters = 1000000
	for i in range(max_iters):
		old_policy_v = compute_policy_v(env, policy, gamma)
		new_policy = extract_policy(env,old_policy_v, gamma)
		if (np.all(policy == new_policy)):
			k=i+1
			break
		policy = new_policy
	return policy,k

def value_iteration(env, gamma):
	v = np.zeros(len(env.P), dtype=np.float64)  # initialize value-function
	max_iters = 100000
	eps = 1e-20
	for i in range(max_iters):
		prev_v = np.copy(v)
		for s in range(len(env.P)):
			q_sa = [sum([p*(r + gamma*prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(len(env.P[0]))] 
			v[s] = max(q_sa)
		if (np.sum(np.fabs(prev_v - v)) <= eps):
			k=i+1
			break
	return v,k
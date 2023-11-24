import numpy as np
from tqdm import tqdm
from utils.callbacks import MyCallbacks
from utils.decorators import print_runtime
import warnings


class RL:
    def __init__(self, env):
        self.env = env
        self.callbacks = MyCallbacks()
        self.render = False

    @staticmethod
    def decay_schedule(init_value, min_value, decay_ratio, max_steps, log_start=-2, log_base=10):
        """
        Parameters
        ----------------------------
        init_value {float}:
            Initial value of the quantity being decayed
        
        min_value {float}:
            Minimum value init_value is allowed to decay to
            
        decay_ratio {float}:
            The exponential factor exp(decay_ratio).
            Updated decayed value is calculated as 
        
        max_steps {int}:
            Max iteration steps for decaying init_value
        
        log_start {array-like}, default = -2:
            Starting value of the decay sequence.
            Default value starts it at 0.01
        
        log_base {array-like}, default = 10:
            Base of the log space.
        
        
        Returns
        ----------------------------
        values {array-like}, shape(max_steps):
            Decay values where values[i] is the value used at i-th step
        """
        decay_steps = int(max_steps * decay_ratio)
        rem_steps = max_steps - decay_steps
        values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
        values = (values - values.min()) / (values.max() - values.min())
        values = (init_value - min_value) * values + min_value
        values = np.pad(values, (0, rem_steps), 'edge')
        return values

    @print_runtime
    def q_learning(self,
                   nS=None,
                   nA=None,
                   convert_state_obs=lambda state, done: state,
                   gamma=.99,
                   init_alpha=0.5,
                   min_alpha=0.01,
                   alpha_decay_ratio=0.5,
                   init_epsilon=1.0,
                   min_epsilon=0.1,
                   epsilon_decay_ratio=0.9,
                   n_episodes=10000):
        """
        Parameters
        ----------------------------
        nS {int}:
            Number of states
        
        nA {int}:
            Number of available actions
            
        convert_state_obs {lambda}:
            The state conversion utilized in BlackJack ToyText problem.
            Returns three state tuple as one of the 280 converted states.
        
        gamma {float}, default = 0.99:
            Discount factor
        
        init_alpha {float}, default = 0.5:
            Learning rate
        
        min_alpha {float}, default = 0.01:
            Minimum learning rate
        
        alpha_decay_ratio {float}, default = 0.5:
            Decay schedule of learing rate for future iterations
        
        init_epsilon {float}, default = 1.0:
            Initial epsilon value for epsilon greedy strategy.
            Chooses max(Q) over available actions with probability 1-epsilon.
        
        min_epsilon {float}, default = 0.1:
            Minimum epsilon. Used to balance exploration in later stages.
        
        epsilon_decay_ratio {float}, default = 0.9:
            Decay schedule of epsilon for future iterations
            
        n_episodes {int}, default = 10000:
            Number of episodes for the agent


        Returns
        ----------------------------
        Q {numpy array}, shape(nS, nA):
            Final action-value function Q(s,a)

        pi {lambda}, input state value, output action value:
            Policy mapping states to actions.

        V {numpy array}, shape(nS):
            State values array

        Q_track {numpy array}, shape(n_episodes, nS, nA):
            Log of Q(s,a) for each episode

        pi_track {list}, len(n_episodes):
            Log of complete policy for each episode
        """
        if nS is None:
            nS=self.env.observation_space.n
        if nA is None:
            nA=self.env.action_space.n
        pi_track = []
        Q = np.zeros((nS, nA), dtype=np.float64)
        Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
        # Explanation of lambda:
        # def select_action(state, Q, epsilon):
        #   if np.random.random() > epsilon:
        #       return np.argmax(Q[state])
        #   else:
        #       return np.random.randint(len(Q[state]))
        select_action = lambda state, Q, epsilon: np.argmax(Q[state]) \
            if np.random.random() > epsilon \
            else np.random.randint(len(Q[state]))
        alphas = RL.decay_schedule(init_alpha,
                                min_alpha,
                                alpha_decay_ratio,
                                n_episodes)
        epsilons = RL.decay_schedule(init_epsilon,
                                  min_epsilon,
                                  epsilon_decay_ratio,
                                  n_episodes)
        for e in tqdm(range(n_episodes), leave=False):
            self.callbacks.on_episode_begin(self)
            self.callbacks.on_episode(self, episode=e)
            state, info = self.env.reset()
            done = False
            state = convert_state_obs(state, done)
            while not done:
                if self.render:
                    warnings.warn("Occasional render has been deprecated by openAI.  Use test_env.py to render.")
                action = select_action(state, Q, epsilons[e])
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                if truncated:
                    warnings.warn("Episode was truncated.  Bootstrapping 0 reward.")
                done = terminated or truncated
                self.callbacks.on_env_step(self)
                next_state = convert_state_obs(next_state,done)
                td_target = reward + gamma * Q[next_state].max() * (not done)
                td_error = td_target - Q[state][action]
                Q[state][action] = Q[state][action] + alphas[e] * td_error
                state = next_state
            Q_track[e] = Q
            pi_track.append(np.argmax(Q, axis=1))
            self.render = False
            self.callbacks.on_episode_end(self)

        V = np.max(Q, axis=1)
        # Explanation of lambda:
        # def pi(s):
        #   policy = dict()
        #   for state, action in enumerate(np.argmax(Q, axis=1)):
        #       policy[state] = action
        #   return policy[s]
        pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
        return Q, V, pi, Q_track, pi_track
    


class Planner:
    def __init__(self, P):
        self.P = P

    @print_runtime
    def value_iteration(self, gamma=1.0, n_iters=1000, theta=1e-10):
        """
        PARAMETERS:

        gamma {float}:
            Discount factor

        n_iters {int}:
            Number of iterations

        theta {float}:
            Convergence criterion for value iteration.
            State values are considered to be converged when the maximum difference between new and previous state values is less than theta.
            Stops at n_iters or theta convergence - whichever comes first.


        RETURNS:

        V {numpy array}, shape(possible states):
            State values array

        V_track {numpy array}, shape(n_episodes, nS):
            Log of V(s) for each iteration

        pi {lambda}, input state value, output action value:
            Policy mapping states to actions.
        """
        V = np.zeros(len(self.P), dtype=np.float64)
        V_track = np.zeros((n_iters, len(self.P)), dtype=np.float64)
        i = 0
        converged = False
        while i < n_iters-1 and not converged:
            i += 1
            Q = np.zeros((len(self.P), len(self.P[0])), dtype=np.float64)
            for s in range(len(self.P)):
                for a in range(len(self.P[s])):
                    for prob, next_state, reward, done in self.P[s][a]:
                        Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
            if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
                converged = True
            V = np.max(Q, axis=1)
            V_track[i] = V
        if not converged:
            warnings.warn("Max iterations reached before convergence.  Check theta and n_iters.  ")
        # Explanation of lambda:
        # def pi(s):
        #   policy = dict()
        #   for state, action in enumerate(np.argmax(Q, axis=1)):
        #       policy[state] = action
        #   return policy[s]
        pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
        return V, V_track, pi

    @print_runtime
    def policy_iteration(self, gamma=1.0, n_iters=50, theta=1e-10):
        """
        PARAMETERS:

        gamma {float}:
            Discount factor

        n_iters {int}:
            Number of iterations

        theta {float}:
            Convergence criterion for policy evaluation.
            State values are considered to be converged when the maximum difference between new and previous state
            values is less than theta.


        RETURNS:

        V {numpy array}, shape(possible states):
            State values array

        V_track {numpy array}, shape(n_episodes, nS):
            Log of V(s) for each iteration

        pi {lambda}, input state value, output action value:
            Policy mapping states to actions.
        """
        random_actions = np.random.choice(tuple(self.P[0].keys()), len(self.P))
        # Explanation of lambda:
        # def pi(s):
        #   policy = dict()
        #   for state, action in enumerate(np.argmax(Q, axis=1)):
        #       policy[state] = action
        #   return policy[s]
        pi = lambda s: {s: a for s, a in enumerate(random_actions)}[s]
        # initial V to give to `policy_evaluation` for the first time
        V = np.zeros(len(self.P), dtype=np.float64)
        V_track = np.zeros((n_iters, len(self.P)), dtype=np.float64)
        i = 0
        converged = False
        while i < n_iters-1 and not converged:
            i += 1
            old_pi = {s: pi(s) for s in range(len(self.P))}
            V = self.policy_evaluation(pi, V, gamma, theta)
            V_track[i] = V
            pi = self.policy_improvement(V, gamma)
            if old_pi == {s: pi(s) for s in range(len(self.P))}:
                converged = True
        if not converged:
            warnings.warn("Max iterations reached before convergence.  Check n_iters.")
        return V, V_track, pi

    def policy_evaluation(self, pi, prev_V, gamma=1.0, theta=1e-10):
        while True:
            V = np.zeros(len(self.P), dtype=np.float64)
            for s in range(len(self.P)):
                for prob, next_state, reward, done in self.P[s][pi(s)]:
                    V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
            if np.max(np.abs(prev_V - V)) < theta:
                break
            prev_V = V.copy()
        return V

    def policy_improvement(self, V, gamma=1.0):
        Q = np.zeros((len(self.P), len(self.P[0])), dtype=np.float64)
        for s in range(len(self.P)):
            for a in range(len(self.P[s])):
                for prob, next_state, reward, done in self.P[s][a]:
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
        # Explanation of lambda:
        # def new_pi(s):
        #   policy = dict()
        #   for state, action in enumerate(np.argmax(Q, axis=1)):
        #       policy[state] = action
        #   return policy[s]
        new_pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
        return new_pi
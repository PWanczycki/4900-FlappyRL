from os.path import isfile
import numpy as np
from ...A2Helpers import *
import matplotlib.pyplot as plt
from queue import PriorityQueue

# credits:
# https://arxiv.org/ftp/arxiv/papers/1206/1206.3285.pdf

class CustomFeaturizer:

    def __init__(self, env):
        self.env = env
        self.n_features = len(env.get_state_shape)

    def featurize(self, state):
        maxes = self.env.get_state_shape
        return np.clip(state / maxes, 0, 1)


def DynaQFA(env, 
            featurizer, 
            gamma=0.99,
            step_size=0.005,
            epsilon=0.1,
            max_episode=400,
            max_model_step=20,
            load_from_file=True):
    start_epsilon = epsilon
    n_actions = 2
    taken_actions = []

    Q = np.zeros((featurizer.n_features, n_actions))
    F = np.zeros((n_actions, featurizer.n_features, featurizer.n_features))
    b = np.zeros((n_actions, featurizer.n_features))

    if load_from_file:
        if isfile("QData.npy"): Q = np.load("QData.npy")
        if isfile("FData.npy"): F = np.load("FData.npy")
        if isfile("BData.npy"): b = np.load("BData.npy")


    def epsilon_greedy_select(s):
        # with small chance, select a random action
        if np.random.uniform() < epsilon:
            a = np.random.choice(n_actions)
        else:
            # otherwise just get the most popular
            a = np.argmax([b[ac].T @ s + gamma * Q.T[ac] @ F[ac] @ s for ac in range(n_actions)])
        if a not in taken_actions: taken_actions.append(a)
        return a
    
    max_reward = None
    all_rewards = []    

    for current_ep in range(max_episode):
        epsilon = start_epsilon / (current_ep * 100 / max_episode + 1) ** 2
        # reset world and get initial state        
        state, _ = env.reset()  
        state = featurizer.featurize(state)
        terminated = truncated = False
        
        total_reward = 0
        while not (terminated or truncated):
            
            # select action
            a = epsilon_greedy_select(state)
            # take action and observe outcome:
            new_state, reward, terminated, truncated, _ = env.step(a)
            total_reward += reward
            new_state = featurizer.featurize(new_state) # _6d(new_state, ts) / 100            

            delta = reward + gamma * (Q.T[a] @ new_state) - Q.T[a] @ state
            Q[:,a] = Q[:,a] + step_size * delta * state             
            F[a] = F[a] + step_size * (new_state - F[a] @ state) * state
            b[a] = b[a] + step_size * (reward - b[a].T @ state) * state
            
            # pqueue = PriorityQueue()
            # for i in range(len(state)):
            #     if state[i] != 0:
            #         pqueue.put((abs(delta * state[i]), i))

            # while pqueue.qsize() > 0:
                
            #     for p in range(max_model_step):                    
            #         i = pqueue.get()[1]
            #         nonzero = [x for x in range(len(state)) if (F[:,i,x] != 0).any()]
            #         print(len(nonzero))
            #         for j in nonzero:
            #             e_j = np.zeros_like(state)
            #             e_j[j] = 1
            #             delta = np.max(b[:,j] + gamma * Q.T @ F[:] @ e_j) - Q.T[j]
            #             Q[j] = Q[j] + step_size * delta
            #             pqueue.put((abs(delta), j))
            # state = new_state
            # continue

            temp = new_state
            
            for p in range(max_model_step):
                state = np.random.sample(featurizer.n_features)
                a = np.random.choice(taken_actions)

                new_state = F[a] @ state                
                r = b[a].T @ state  
                
                Q[:,a] = Q[:,a] + step_size * (r + (gamma * Q.T[a] @ new_state) - (Q.T[a] @ state)) * state                
                                
            state = temp
            
        if not max_reward or max_reward < total_reward:
            max_reward = total_reward
        all_rewards.append(total_reward)

    # Pi = np.zeros_like(Q)
    
    # for i in range(len(Q)):
    #     Pi[i, np.argmax(Q[i])] = 1

    # Pi = diagonalization(Pi, env.n_states, env.n_actions)
    print(f"maximum reward: {max_reward}")
    np.save("rewards.npy", all_rewards)
    plt.plot(all_rewards)
    plt.show()
    return Q, F, b # Pi, np.reshape(Q, (env.n_states * env.n_actions, 1))

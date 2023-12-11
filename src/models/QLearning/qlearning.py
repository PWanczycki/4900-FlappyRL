import numpy as np
from ..A2Helpers import *
from matplotlib import pyplot as plt

class QLearner:
    def run(self, env, gamma, step_size, epsilon, max_episode, callback_step, callback):

        plt.figure()

        def epsilon_greedy_select(q):
            # with small chance, select a random action
            if np.random.uniform() < epsilon:
                return np.random.randint(len(q))
            else:
                # otherwise just get the most popular
                return np.argmax(q)

        reduced_shape = (10,5,10,15,10,15)
        reduction_multipliers = [reduced_shape[i] / env.get_state_shape[i] for i in range(len(reduced_shape))]
        Q = np.zeros(reduced_shape + (env.n_actions,))

        scores = []

        for episode in range(max_episode):

            # reset world and get initial state        
            obs, _ = env.reset()

            state = [int(obs[i] * reduction_multipliers[i]) for i in range(len(obs))]

            terminated = False
            score = 0
            while not terminated:
                
                # select action
                # print(state)
                action = epsilon_greedy_select(self._lookup(Q, state))
                
                # take action and observe outcome:
                obs, reward, terminated, _, _ = env.step(action)
                score += 1 

                # update Q value
                new_state = state = [int(obs[i] * reduction_multipliers[i]) for i in range(len(obs))]
                # print(new_state)
                self._lookup(Q,state)[action] = self._lookup(Q, state)[action] + step_size * (reward + gamma * max(self._lookup(Q,new_state)) - self._lookup(Q, state)[action])
            
                # update state
                state = new_state

            scores.append(score)

        plt.plot(scores)
        plt.show()
        Q_2D = np.reshape(Q, (-1, 2))
        Pi = np.zeros_like(Q_2D)
        
        for i in range(len(Q_2D)):
            Pi[i, np.argmax(Q_2D[i])] = 1

        # Pi = diagonalization(Pi, np.prod(list(env.get_state_shape)), env.n_actions)

        return Pi, Q_2D # np.reshape(Q_2D, (np.prod(list(env.get_state_shape)) * env.n_actions, 1))

    def _lookup(self, data, indexers):
        if len(indexers) == 0:
            return data
        
        return self._lookup(data[indexers[0]], indexers[1:])
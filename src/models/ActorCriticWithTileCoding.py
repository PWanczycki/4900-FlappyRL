
import numpy as np
import matplotlib.pyplot as plt 
import random
import scipy.special
import math

#Q3a
def softmaxProb(x,Theta):
    h_S = Theta.T@x #(3,)
    max_h = np.max(h_S) 
    new_h = h_S-max_h #(3,)
    exp = np.exp(new_h)
    sum = np.sum(exp)
    pi = (np.exp(new_h))/sum
    return pi

#Q3b 
def softmaxPolicy(x, Theta):
    pi = softmaxProb(x, Theta)
    a = np.random.choice(np.arange(len(pi)),p=pi)
    return a


#Q3c 
def logSoftmaxPolicyGradient(x, a, Theta):
    d = np.shape(x)[0]
    k = np.shape(Theta)[1]
    x_S_in_A_column = np.zeros((d,k))
    x_S_in_A_column[:,a] = x #change this to the one below when running jax
    #x_S_in_A_column = x_S_in_A_column.at[:,a].set(x)
    pi = softmaxProb(x,Theta)
    repeated_x_S_row = np.tile(x, (k, 1))
    repeated_x_S = repeated_x_S_row.T
    x_S_weighted_by_pi = repeated_x_S * pi
    logGradient = x_S_in_A_column - x_S_weighted_by_pi
    return logGradient

#Q3d 
def ActorCritic(env,featurizer,eval_func,gamma=0.99,actor_step_size=0.005,critic_step_size=0.005,max_episodes=3000,evaluate_every=20):

    n_actions = env.action_space.n
    n_features = featurizer.n_features

    Theta = np.random.rand(n_features,n_actions)
    w = np.random.rand(n_features)
    
    eval_returns = [] #added this line

    rewards = []
    
    

    for i in range(1, max_episodes + 1):
        episode_Score = 0 #added this line
        s, info = env.reset()
        s = featurizer.featurize(s)
        terminated = truncated = False
        actor_discount = 1
        while not (terminated or truncated):
            # TODO: Compute essential quantities, update Theta and w
            # and proceed to next state of env
            
            action = softmaxPolicy(s,Theta)
            next_state, reward, terminated, truncated, info = env.step(action)
            delta = reward + gamma*(np.dot(w.T,featurizer.featurize(next_state)))-np.dot(w.T,s)
            episode_Score += reward #added this line
            w += critic_step_size*delta*s
            Theta += actor_step_size*delta*actor_discount*logSoftmaxPolicyGradient(s,action,Theta)

            actor_discount *= gamma

            s = featurizer.featurize(next_state)
        rewards.append(episode_Score)

        if i % evaluate_every == 0:
            eval_return = eval_func(env, featurizer, Theta, softmaxPolicy)
            eval_returns.append(eval_return)
            print(eval_return)

   
    fig, ax = plt.subplots() #added this line
    line, = ax.plot(rewards) #added this line
    plt.xlabel('Episode') #added this line
    plt.ylabel('Reward') #added this line
    # plt.savefig('AC-with-TC.png') #added this line

    print(rewards)
    plt.show()
    
        
    return Theta, w, eval_returns
    
def evaluate(env, featurizer, W, policy_func, n_runs=10):
    all_returns = np.zeros([n_runs])
    for i in range(n_runs):
        observation, info = env.reset()
        return_to_go = 0
        while True:
            observation = featurizer.featurize(observation)
            action = policy_func(observation, W)

            observation, reward, terminated, truncated, info = env.step(action)
            return_to_go += reward
            if terminated or truncated:
                break
        all_returns[i] = return_to_go

    return np.mean(all_returns) 
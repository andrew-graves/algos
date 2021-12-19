# -*- coding: utf-8 -*-
"""
Various temporal-difference learning routines
Andrew J. Graves
04/02/21
"""

# Import modules
import numpy as np

# TD(0) algorithm
def td_0(get_episode, policy, initial_v, gamma, alpha, num_episodes=1):
# This function implements TD(0).

    # Initialize value function
    v = np.copy(initial_v)
    
    for ep in range(int(num_episodes)):
        states, _, rewards = get_episode(policy) # generate an episode
        
        # Iterate through all non-terminal states
        for idx, s in enumerate(states[:-1]):
            # Store the next state
            sp = states[idx+1]
            # Do one-step lookahead TD learning
            v[s] += alpha*(rewards[idx] + gamma*v[sp] - v[s])
    
    return v

# n-step TD algorithm
def td_n(get_episode, policy, initial_v, n, gamma, alpha, num_episodes=1):

    # Initialize value function
    v = np.copy(initial_v)
    
    for ep in range(int(num_episodes)):
        states, _, rewards = get_episode(policy) # generate an episode
        
        # Initialize T, tau, and t (counter)
        T = np.inf
        tau = 0
        t = 0
        
        # Iterate through the episode
        while tau != T-1:
            # If the next state is the terminal state
            if t < T and states[t+1] == states[-1]:
                # Update the value of T for computing G
                T = t+1
            # Subtract n from current iteration index and add 1 to get tau
            tau = t-n+1 
            # If tau is a valid index
            if tau >= 0:
                # Accumulate returns from n-steps
                G = np.sum([gamma**(i-tau)*rewards[i] 
                                for i in range(tau, min(tau+n, T))])
                if tau+n < T:
                    G += gamma**n*v[states[tau+n]]
                
                # Update the value at tau
                v[states[tau]] += alpha*(G-v[states[tau]])
            # Increment the iterator
            t += 1

    return v

# Backward-view TD($\lambda$) algorithm
def td_lambda(get_episode, policy, initial_v, lambda_, gamma, alpha,
              num_episodes=1):

    # Initialization
    v = np.copy(initial_v) # value function

    for ep in range(int(num_episodes)):
        states, _, rewards = get_episode(policy) # generate an episode
        # Initialize eligibility traces to 0
        e = np.zeros(len(v))

        # Iterate through all non-terminal states
        for idx, s in enumerate(states[:-1]):
            # Store the next state
            sp = states[idx+1]

            # Compute the global TD error signal
            delta = rewards[idx] + gamma*v[sp] - v[s]
            # Update and accumulate eligibility traces
            e *= gamma*lambda_
            e[s] += 1
            # Update the value function
            v += alpha*delta*e
            
    return v
# -*- coding: utf-8 -*-
"""
Several useful dynamic programming routines for RL problems
Andrew J. Graves
03/18/21
"""

# Import modules
import numpy as np

# Synchronous policy evaluation algorithm
def policy_eval(policy, P, R, gamma, theta, max_iter=1e8):

    # Initialization 
    num_S, num_a = policy.shape # get state and action counts   
    v = np.zeros(num_S) # value function
    k = 0 # counter of iteration
    
    # Start delta > theta
    delta = 1
    # Iterate until solution has converged
    while delta > theta and k < max_iter:
        # Reset delta to 0
        delta = 0
        
        # Store old state values
        old_v = v.copy()
        # Begin state sweep
        for s in range(num_S):
            # Initialize action value storage
            av = []
            # Consider all possible actions
            for a in range(num_a):
                # Get next states (s')
                sp = np.where(P[s][a])
                # Compute the Bellman optimality equation
                av.append(policy[s][a] * np.sum(P[s][a][sp] * (R[s][a][sp] + gamma*old_v[sp])))

            # Update value of the current state
            v[s] = np.sum(av)
            
            # Update convergence test
            delta = max(delta, abs(old_v[s]-v[s]))
        # Update counter
        k += 1
    
    return v
    
# Policy improvement algorithm.
def policy_imprv(P, R, gamma, policy, v):

    # Initialization    
    num_S, num_a = policy.shape # get state and action counts
    policy_new = np.zeros([num_S,num_a]) # start policy at 0 everywhere
    policy_stable = True # start with stable policy
    
    # Begin state sweep
    for s in range(num_S):
        # Store the old policy
        old_action = policy[s]
        # Initialize action value storage
        av = []
        # Consider all possible actions
        for a in range(num_a):
            # Get next states (s')
            sp = np.where(P[s][a])
            # Compute the Bellman optimality equation
            av.append(np.sum(P[s][a][sp] * (R[s][a][sp] + gamma*v[sp])))
            
        # Take action with largest value
        # Break ties arbitrarily (take earliest max index)
        policy_new[s] = np.eye(num_a)[np.argmax(av)]

        # If the old and new policy are not the same
        if not np.array_equal(old_action, policy_new[s]):
            policy_stable = False
    
    return policy_new, policy_stable

# Policy iteration algorithm.
def policy_iter(P, R, gamma, theta, initial_policy, max_iter=1e6):

    policy_stable = False
    policy = np.copy(initial_policy)
    num_iter = 0
    
    while (not policy_stable) and num_iter < max_iter:
        num_iter += 1
        print('Policy Iteration: ', num_iter)
        # policy evaluation
        v = policy_eval(policy,P,R,gamma,theta)
        # policy improvement
        policy, policy_stable = policy_imprv(P,R,gamma,policy,v)
    return policy, v

# In-place value iteration algorithm
def value_iter(P, R, gamma, theta, initial_v, max_iter=1e8):  
    
    # Initialization
    v = initial_v # value function
    num_S, num_a = P.shape[:2] # get state and action counts
    best_actions = [0] * num_S # start actions at 0
    k = 0 # counter of iteration
    
    # Start delta > theta
    delta = theta + 1
    # Iterate until solution has converged
    while delta > theta and k < max_iter:
        # Reset delta to 0
        delta = 0
        
        # Begin state sweep
        for s in range(num_S):
            # Assign the previous value function
            old_v = v[s]
            # Initialize action value storage
            av = []
            # Consider all possible actions
            for a in range(num_a):
                # Get next states
                sp = np.where(P[s][a])
                # Compute the Bellman optimality equation
                av.append(np.sum(P[s][a][sp] * (R[s][a][sp] + gamma*v[sp])))

            # Update value of the current state
            v[s] = np.max(av)
            # Get best actions
            best_actions[s] = np.argmax(av)
            
            # Update convergence test
            delta = max(delta, abs(old_v-v[s]))
            
        # Update counter
        k += 1
    
    print('number of iterations:', k)
    return best_actions, v
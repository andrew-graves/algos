# -*- coding: utf-8 -*-
"""
State-action-reward-state-action (SARSA) reinforcement learning routines
Andrew J. Graves
04/20/21
"""

# Import modules
import numpy as np

# Traditional SARSA algorithm
def sarsa(initial_Q,initial_state,transition,
          num_episodes, gamma, alpha, epsilon=0.1):

    # Initialization    
    Q = np.copy(initial_Q)
    num_states, num_actions = Q.shape   
       
    # Store number of steps in each episode
    steps = np.zeros(num_episodes,dtype=int)
    # Store total rewards for each episode
    rewards = np.zeros(num_episodes)
    
    # Iterate through the episodes
    for ep in range(num_episodes):
        # Get initial state
        s = initial_state
        
        # Initialize action epsilon-greedily
        if np.random.random() < epsilon:
            a = np.random.choice(num_actions)
        else:
            a = np.argmax(Q[s])
            
        # Start at a non-terminal state
        terminal = False
        # Continue until the terminal state
        while not terminal:
            # Transition to the next state and receive reward
            sp, r, terminal = transition(s,a)
            
            # Act epsilon greedy w.r.t policy
            if np.random.random() < epsilon:
                ap = np.random.choice(num_actions)
            else:
                ap = np.argmax(Q[sp])
                
            # Update Q
            Q[s,a] += alpha*(r + gamma*Q[sp,ap] - Q[s,a])
            # Assign next state-action to current state-action
            s, a = sp, ap
            
            # Update rewards and steps within each episode
            rewards[ep] += r
            steps[ep] += 1
            
    return Q, steps, rewards

# SARSA $\lambda$ algorithm
def sarsa_lambda(initial_Q,initial_state,transition,
          num_episodes,gamma, alpha, lambda_, epsilon=0.1):
    
    # Initialization
    Q = np.copy(initial_Q)
    num_states, num_actions = Q.shape   
    
    # Store number of steps in each episode
    steps = np.zeros(num_episodes,dtype=int)
    # Store total rewards for each episode
    rewards = np.zeros(num_episodes)
    
    # Iterate through the episodes
    for ep in range(num_episodes):
        # Zero out eligibility traces
        e = np.zeros([num_states, num_actions])
        # Get initial state
        s = initial_state
        
        # Initialize action epsilon-greedily
        if np.random.random() < epsilon:
            a = np.random.choice(num_actions)
        else:
            a = np.argmax(Q[s])
        
        # Start at a non-terminal state
        terminal = False
        # Continue until the terminal state
        while not terminal:
            # Transition to the next state and receive reward
            sp, r, terminal = transition(s,a)
            
            # Act epsilon greedy w.r.t policy
            if np.random.random() < epsilon:
                ap = np.random.choice(num_actions)
            else:
                ap = np.argmax(Q[sp])
            
            # Compute delta
            delta = r + gamma*Q[sp,ap] - Q[s,a]
            # Increment eligiblity traces
            e[s,a] += 1
            # Iterate through each state-action pair
            for s in range(num_states):
                for a in range(num_actions):
                    # Update Q
                    Q[s,a] += alpha*delta*e[s,a]
                    # Update eligibility trace
                    e[s,a] = gamma*lambda_*e[s,a]
            # Assign next state-action to current state-action
            s, a = sp, ap
            
            # Update rewards and steps within each episode
            rewards[ep] += r
            steps[ep] += 1
            
    return Q, steps, rewards
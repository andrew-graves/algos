# -*- coding: utf-8 -*-
"""
Q-learning reinforcement learning routines
Andrew J. Graves
04/24/21
"""

# Import modules
import numpy as np

# Q-learning algorithm
def q_learn(initial_Q, initial_state, transition,
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
            
        # Start at a non-terminal state
        terminal = False
        # Continue until the terminal state
        while not terminal:
            
            # Initialize action epsilon-greedily
            if np.random.random() < epsilon:
                a = np.random.choice(num_actions)
            else:
                a = np.argmax(Q[s])
                
            # Transition to the next state and receive reward
            sp, r, terminal = transition(s,a)
                
            # Update Q
            Q[s,a] += alpha*(r + gamma*np.max(Q[sp]) - Q[s,a])
            # Assign next state to current state
            s = sp
            
            # Update rewards and steps within each episode
            rewards[ep] += r
            steps[ep] += 1
            
    return Q, steps, rewards

# Double Q-learning
def doubleQ(initial_Q1, initial_Q2, initial_state, transition,
            num_episodes, gamma, alpha, epsilon=0.1):
    
    # Initialization    
    Q1 = np.copy(initial_Q1)
    Q2 = np.copy(initial_Q2)
    num_states, num_actions = Q1.shape   
    
    # Iterate through the episodes
    for ep in range(num_episodes):
        # Get initial state
        s = initial_state
            
        # Start at a non-terminal state
        terminal = False
        # Continue until the terminal state
        while not terminal:
            
            # Initialize action epsilon-greedily
            if np.random.random() < epsilon:
                a = np.random.choice(num_actions)
            else:
                a = np.argmax(Q1[s] + Q2[s])
                
            # Transition to the next state and receive reward
            sp, r, terminal = transition(s,a)
            
            # Coin-flip for updating Q1 or Q2
            if np.random.random() < 0.5:
                # Update Q1
                Q1[s,a] += alpha*(r + gamma*Q2[sp, np.argmax(Q1[sp])] - Q1[s,a])
            else:
                # Update Q2
                Q2[s,a] += alpha*(r + gamma*Q1[sp, np.argmax(Q2[sp])] - Q2[s,a])

            # Assign next state to current state
            s = sp
            
        # Return the final state-action value function
        Q = np.mean([Q1, Q2], axis=0)
    
    return Q1, Q2, Q
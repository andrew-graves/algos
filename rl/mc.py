# -*- coding: utf-8 -*-
"""
Monte-Carlo reinforcement learning routines
Andrew J. Graves
04/02/21
"""

# Import modules
import numpy as np

# Monte Carlo every-visit algorithm
def mc_evisit(get_episode, policy, initial_v, gamma, alpha, num_episodes=1):

    # Initialization  
    num_states = policy.shape[0] # number of states
    v = np.copy(initial_v) # value function
    N_s = np.zeros(num_states) # counter for visits to states
   
    for ep in range(int(num_episodes)):
        states, _, rewards = get_episode(policy) # generate an episode
        
        # Initialize G at 0
        G = 0
        # Go backwards through each step of the episode
        for step in range(len(states)-1, -1, -1):
            # Get state index
            s = states[step]
            
            # Assumes there is a reward associated with each state
            try:
                r = rewards[step]
            # Handles if len of reward vector < len of state vector
            except:
                r = 0
            
            # Update G
            G = gamma*G + r
            # Update state counter
            N_s[s] += 1
            if alpha == 0:
                # Compute expectation of the value function
                v[s] += (G - v[s]) / N_s[s]
            else:
                # Update the value function with a constant learning rate
                v[s] += alpha * (G - v[s])
     
    return v

# Monte Carlo exploring-starts algorithm
def mc_es(get_episode, initial_Q, initial_policy,
          gamma, alpha, num_episodes=1e4):

    # Initialization  
    Q = np.copy(initial_Q)
    policy = np.copy(initial_policy)
    num_states, num_actions = Q.shape
    N_sa = np.zeros([num_states,num_actions]) # counter of (s,a)
    
    iteration = 0
    
    # Continue through the specified episode count
    while iteration < num_episodes:
        
        # Instantiate the exploring starts
        init_s = np.random.randint(0, num_states)
        init_a = np.random.randint(0, num_actions)
        
        # Generate an episode with exploring starts
        states, actions, rewards = get_episode(policy, init_s, init_a) 
        # Initialize G at 0
        G = 0
        # Go backwards through each step of the episode
        for step in range(len(states)-1, -1, -1):
            # Get state, action, and reward at current index
            s = states[step]
            a = actions[step]
            r = rewards[step]
            
            # Only update the first-visit state-action pairs
            update_Q = True
            for prev in range(step-1, -1, -1):
                if states[prev] == s and actions[prev] == a:
                    update_Q = False
             
            # Update G
            G = gamma*G + r
            if update_Q:
                # Update state-action counter
                N_sa[s,a] += 1
                if alpha == 0:
                    # Compute expectation of the Q-value function
                    Q[s,a] += (G - Q[s,a]) / N_sa[s,a]
                else:
                    # Update the Q-value function with a constant learning rate
                    Q[s,a] += (G - Q[s,a]) * alpha
                
                # Update the policy
                policy[s] = np.eye(num_actions)[np.argmax(Q[s])]
        
        iteration += 1
        
    return Q, policy
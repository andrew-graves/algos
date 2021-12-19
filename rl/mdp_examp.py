# -*- coding: utf-8 -*-
"""
Example of Markov Reward / Decision Process
Andrew J. Graves
03/02/21
"""

# Import modules
import numpy as np

def gen_mrp(n_episodes=3, init_state='C1'):

    # Specify state space
    S = np.array(['C1', 'C2', 'C3', 'Pass', 'Pub', 'FB', 'Sleep'])

    # Specify rewards associated with each state
    R = np.array([-2, -2, -2, 10, 1, -1, 0])

    # Build state-transition probability matrix
    P = np.array([[ 0, .5,  0,  0,  0, .5,  0],
                  [ 0,  0, .8,  0,  0,  0, .2],
                  [ 0,  0,  0, .6, .4,  0,  0],
                  [ 0,  0,  0,  0,  0,  0,  1],
                  [.2, .4, .4,  0,  0,  0,  0],
                  [.1,  0,  0,  0,  0, .9,  0],
                  [ 0,  0,  0,  0,  0,  0,  1]])
    
    # Initialize empty episode list
    episodes = []
    # Generate episodes
    for i in range(n_episodes):
        
        # Get the initial state
        s = S[np.squeeze(np.where(init_state == S))]
        # Initialize return at 0
        r = 0
        # Set up storage for state-reward tuples
        e = []

        # Continue until agent reaches sleep state
        while s != S[-1]:

            # Get index of current state
            idx = np.where(s == S)

            # Store the state and reward
            e.append((s, R[idx].item()))
            # Traverse to the next state
            s = np.random.choice(S, p=np.squeeze(P[idx]))
            # Accumulate return
            r += R[idx]

        # Add sleep state to episode
        e.append((S[-1], 0))
        print(f'-----Episode {i+1} (Return = {r.item()})-----\n\n {e}\n')
        episodes.append(e)
    return episodes

# Generate 3 episodes of MRP
np.random.seed(49)
gen_mrp()

# Function for generating episode samples from the undiscounted student MDP
def gen_mdp(n_episodes=3, init_state='C1', prob_study=.7):
    
    # Specify action space
    A = np.array(['Study', 'Relax'])
    
    # Specify state space
    S = np.array(['C1', 'C2', 'C3', 'Pass', 'Pub', 'FB', 'Sleep'])
    
    # Specify rewards associated with each state
    R = np.array([-2, -2, -2, 10, 1, -1, 0])
    
    # Which states involve relaxing
    is_relax = np.array([0, 0, 0, 0, 1, 1, 1], dtype='bool')
    
    # Build state-transition adjacency matrix (with pub as a stochastic transition)
    P = np.array([[ 0,  1,  0, 0, 0, 1, 0],
                  [ 0,  0,  1, 0, 0, 0, 1],
                  [ 0,  0,  0, 1, 1, 0, 0],
                  [ 0,  0,  0, 0, 0, 0, 1],
                  [.2, .4, .4, 0, 0, 0, 0],
                  [ 1,  0,  0, 0, 0, 1, 0],
                  [ 0,  0,  0, 0, 0, 0, 1]])
    
    # Initialize empty episode list
    episodes = []
    # Generate episodes
    for i in range(n_episodes):
        
        # Get the initial state
        s = S[np.squeeze(np.where(init_state == S))]
        # Initialize return at 0
        r = 0
        # Set up storage for state-action-reward tuples
        e = []
        
        # Continue until agent reaches sleep state
        while s != S[-1]:
            
            # Get index of current state
            idx = np.where(s == S)
            
            # Randomly sample the action (stochastic policy)
            a = np.random.choice(A, p=[prob_study, 1-prob_study])
            
            # Check to see if relaxing is NOT an option
            if not np.sum(P[idx] * is_relax):
                # Only option is to study
                a = A[0]
            # Check to see if studying is NOT an option
            elif not np.sum(P[idx] * np.invert(is_relax)):
                # Only option is to relax
                a = A[1]
            
            # If action is to study
            if a == A[0]:
                # Mask all relaxing states
                s_p = P[idx] * np.invert(is_relax)
            # If action is to relax
            else:
                # Mask all studying states
                s_p = P[idx] * is_relax
                
            # If we reach the Pub
            if s == S[4]:
                a = 'No Action'
                
            # Store the state, action, and reward
            e.append((s, a, R[idx].item()))
                
            # Traverse to the next state
            s = np.random.choice(S, p=np.squeeze(s_p)/s_p.sum())
            # Accumulate return
            r += R[idx]
            
        # Add sleep state to episode
        e.append((S[-1], A[1], 0))
        print(f'-----Episode {i+1} (Return = {r.item()})-----\n\n {e}\n')
        episodes.append(e)
    return episodes

# Generate 3 episodes of MDP
np.random.seed(49)
gen_mdp()
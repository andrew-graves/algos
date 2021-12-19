# -*- coding: utf-8 -*-
"""
Greedy, approximate, and dynamic programming approach to 0-1 knapsack problem
Andrew J. Graves
10/13/20
"""

# Import modules
import numpy as np

# Greedy 0-1 knapsack solution with value-to-weight ratio heuristic
def ks_greedy(vals, wts, cap):
    
    # Initialize item indices and result values
    items = []
    values = []
    
    # Initialize weight and best value
    weight = 0
    best = 0
    
    # Calculate value-to-weight ratio and return indices in decreasing order
    greedy_idx = np.argsort(-1 * np.divide(v, w)) # function of nlgn
    
    # Iterate through the value-to-weight ratio indices
    # Using 'for' instead of 'while' to catch potential light values after initial overflow
    for i in greedy_idx: # function of n
        
        # Checks if there is a single element that beats any set
        if vals[i] > best and wts[i] <= cap:
            best = vals[i]
            best_idx = i
        # Add sorted value-to-weight ratios if they fit
        if weight + wts[i] <= cap:
            items.append(i)
            values.append(vals[i])
            weight += wts[i]
    
    # Return the value sum and the value indices
    value_sum = np.sum(values)
    
    # Check if best single value beats set of values
    if best > value_sum:
        value_sum = best
        items = best_idx

    print(f'Value sum: {value_sum}')
    print(f'Value indices: {items}')
    return value_sum, items

# Dynamic programming solution to the 0-1 knapsack problem
def ks_dyn(vals, wts, cap, large_vals=None):
    
    # Initialize item x capacity matrix
    rows = len(vals) + 1 # augment with null item
    cols = cap + 1 # augment with capacity 0
    arr = np.zeros([rows, cols])
    
    # Initialize item recovery (which vals / wts)
    items = []
    final_vals  = []
    res = arr.copy()

    # Iterate through input items (vals / wts)
    for i in range(1, rows):
        # Iterate through possible capacities (starting at 1)
        for w in range(1, cols):

            # If current weight candidate fits and adds value to current knapsack
            if wts[i-1] <= w and vals[i-1] + arr[i-1, w - wts[i-1]] > arr[i-1, w]:
                arr[i, w] = vals[i-1] + arr[i-1, w - wts[i-1]] # store the new value
                res[i, w] = 1 # mark for item recovery later
            else:
                arr[i, w] = arr[i-1, w] # store the old value
                res[i, w] = 0 # item should not be recovered
    
    # Iterate through the flags in reverse to recover the items
    for i in range(rows-1, -1, -1):
        
        # Decrement column search by weights at flag indices
        if res[i, cap] == 1:
            items.append(i-1)
            cap -= wts[i-1]
            try:
                final_vals.append(large_vals[i-1])
            except:
                pass
                
    # Return the value sum and the value indices
    if large_vals:
        value_sum = sum(final_vals)
    else:
        value_sum = int(arr[rows-1, cols-1])
    print(f'Value sum: {value_sum}')
    print(f'Value indices: {items}')
    return value_sum, items

# Approximation with shrinkage parameter epsilon
def ks_approx(vals, wts, cap, eps=.5):
    
    # Initialize small values
    small_vals = []
    # Compute delta to scale down the values
    delta = (eps * max(vals)) / len(vals)
    
    for i in range(len(vals)):
        # Convert to integer and scale down values
        small_vals.append(np.floor(vals[i] / delta))
    
    # Run the dynamic programming routine with scaled down values
    return ks_dyn(small_vals, wts, cap, vals)

# Simple test of the greedy algorithm
v = [1, 4, 3, 5, 100] 
w = [5, 4, 6, 3, 40] 
c = 10

greedy_res = ks_greedy(v, w, c)

# Tests for approximation and DP
v = [12, 1, 6, 2, 10] 
w = [1, 2, 3, 1, 10] 
c = 10

# No worse than .1 * 21
print('Least optimal (large epsilon)\n---')
ks_approx(v, w, c, eps=.9)

# No worse than .5 * 21
print('\nApproximately optimal (medium epsilon)\n---')
ks_approx(v, w, c, eps=.5)

print('\nOptimal\n---')
opt_res = ks_dyn(v, w, c)
# -*- coding: utf-8 -*-
"""
Randomized Selection Algorithm
Andrew J. Graves
9/30/20
"""

# Import modules
import numpy as np
from random import randrange

# The notation for the following routines follows the CLRS description

# A: the input array
# p: the starting index for partitioning
# r: the end index for partitioning
# i: the ith order statistic

# Partition
def part(A, p, r):
    
    # Define the pivot
    x = A[r]
    # Set i to the beginning of the partition range
    i = p
    # Iterate from the beginning to the end
    for j in range(p, r):
        # If a value is less than the pivot
        if A[j] <= x:
            # Swap ith element with jth element
            A[i], A[j] = A[j], A[i]
            i += 1
    # Swap the jth element with the pivot
    A[i], A[r] = A[r], A[i]
    return i

# Randomized partition
def rand_part(A, p, r):
    # Randomly sample a pivot from the indices from A
    i = randrange(p, r)
    # Place pivot at end for partition recurision
    A[i], A[r] = A[r], A[i]
    return part(A, p, r) # Call partition routine with random pivot

# Randomized selection
def rand_select(A, p, r, i):
    
    # Handles when i too small or too large
    if i < 0 or i > len(A):
        print(f'{i} is out of bounds. Please select another j-th order statistic.')
        return
    # Handles when p / r is too small / large
    if p < 0 or r > len(A) - 1 or p > r:
        print('Selection region is out of bounds for the input array')
        return
    # Handles 1 element case
    if p == r:
        return A[p]
    
     # Call random partition routine to update p / r / i for recursion
    q = rand_part(A, p, r)
    # Adaptively shifts ith order input appropriately
    k = q - p + 1
    if i == k:
        # Select result
        return A[q]
    elif i < k:
        # Recurisvely select earlier portion of array
        return rand_select(A, p, q-1, i)
    else:
        # Recurisvely select later portion of array
        return rand_select(A, q+1, r, i - k)
    
# Generate random test array
test_array = np.random.randint(-10, 10, size=2000)

# Specify starting value (counting from 1)
start = 1
# Specify ending value (bounded by list length)
stop = len(test_array)

failures = 0
# Iterate over all j-th orders for this arbitrary test array
for j in range(1, len(test_array)):
    # Ensure random selection returns same result as indexing on a sorted array
    if not rand_select(test_array, start-1, stop-1, j) == np.sort(test_array)[j-1]:
        print('Random testing failed!')
        failures += 1
if failures == 0:
    print('Passed all random tests!')
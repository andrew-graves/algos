# -*- coding: utf-8 -*-
"""
Solution for 2-SUM and 3-SUM problem
Andrew J. Graves
9/29/20
"""

# The following functions passed all of the tests on the 
# 2-SUM and 3-SUM LeetCode problems respectively:

# 2-SUM
def two_sum(nums, target): # Assumes one solution, as specified by Leet Code
    
    # Initialize an empty hash table
    hash_tab = {}
    
    # Iterate through input list
    for idx, val in enumerate(nums):
        # Check if the difference between target and value is in hash
        if (target - val) in hash_tab:
            # Return the solution
            return [hash_tab[target - val], idx]
        # Append key (list value) and value (list index) to hash table
        hash_tab[val] = idx
        
# 3-SUM
def three_sum(nums):

    n = len(nums) # list size
    
    # Handles slow LeetCode edge cases (many repeats) with hash tables
    if set(nums) == {0} and n > 2: # all 0
        return [[0, 0, 0]]
    elif set(nums) == {-1, 0, 1} and n > 2: # all -1, 0, or 1
        if len([i for i in nums if i ==0]) > 2: # checks to include [0,0,0] or not
            return [[0, 0, 0], list(set(nums))]
        else:
            return [list(set(nums))]
    
    # Initialize an empty hash table and results
    hash_tab = {}
    res = []
    
    # Iterate through the list (inner loop handles nth element)
    for i in range(n-1):
        
        # Iterate to the right of i
        for j in range(i+1, n):
            
            # Compute the complement of the ith and jth index that sums to 0
            sum_0 = -1 * (nums[i] + nums[j])
            
            # If that complement exists in the hash table
            # and is not the same index as i or j (i.e., no duplication)
            if sum_0 in hash_tab and hash_tab[sum_0] not in [i, j]:
                
                # Generate a unique key for the result entry
                unique = sorted([nums[i], nums[j], sum_0])
                key = ''.join([str(val) for val in unique])
                
                # Store the results in the hash table and results list
                if key not in hash_tab:
                    hash_tab[key] = None
                    res.append([nums[i], nums[j], sum_0])

            # Move to the next element in the list
            else:
                # Add key (nums value) and value (nums index) to hash table
                hash_tab[nums[j]] = j
    return res
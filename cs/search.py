# -*- coding: utf-8 -*-
"""
Simple linear and binary search implementation
Andrew J. Graves
7/19/20
"""

# Import modules
import sys
import os

# Linear search algorithm w/ sentinel value = -1
def lin_search(target, item_list):
    
    # Loop through elements of the list
    for idx in range(len(item_list)):
        # Return index if match is found
        if item_list[idx] == target:
            return idx
    # Return sentinel value if no match is found
    return -1

# Binary search algorithm; allows for unsorted lists
def bin_search(target, item_list, is_sorted=False):
    
    # Initialize start and end point
    start = 0
    end = len(item_list) - 1
    
    # Checks if user asked list to be sorted
    if is_sorted == False:
        sort = sorted(item_list)
    else:
        sort = item_list
    
    while start <= end:
        
        # Compute current mid-point
        mid = (start + end) // 2
        print(f'Mid is: {mid}')
        
         # Returns True if match is found
        if sort[mid] == target:
            return True
        # Update start and end if match not found
        elif target > sort[mid]:
            start = mid + 1
        else:
            end = mid - 1
    
    # Returns False if no match is found
    return False


# Linear search example list
lin_test_list = ['lion', 'tiger', 'elephant', 'zebra', 'bear']

# Example test
for i in ['tiger', 'zebra', 'panda']:
    if lin_search(i, lin_test_list) != -1:
        print(f'{i} is at index {lin_search(i, lin_test_list)}')
    else:
        print(f'{i} was not found. Returned {lin_search(i, lin_test_list)}')

# Binary search example list
bin_test_list = [3, 5, 2, 1, 8]

# Example test
for i in [6, 5, 1]:
    print(f'\n----- Searching for {i} -----\n')
    print(f'Is {i} in {bin_test_list}? {bin_search(i, bin_test_list)}')
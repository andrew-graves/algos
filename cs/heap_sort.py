# -*- coding: utf-8 -*-
"""
Simple heapsort implementation
Andrew J. Graves
10/13/20
"""

# Import modules
import numpy as np

# Bare-bones heap data structure with methods relevant for heapsort
class Heap:
    
    # The notation for the following methods follows the CLSR description (Ch 6)
    
    # Constructor for Heap class
    def __init__(self, arr):
        # Specify array as A
        self.A = arr
        # Get the heapsize
        self.heapsize = len(arr)
     
    # Get left-child at node i
    def left(self, i):
        self.l = 2*i + 1
    
    # Get right-child at node i
    def right(self, i):
        self.r = 2*i + 2
    
    # Update heap size after 'pop'
    def update_size(self):
        self.heapsize = self.heapsize - 1
        
    # Get a (max)-heap with dynamic index i
    def heapify(self, i):

        # Set left-child with respect to node i
        self.left(i)
        # Set right-child with respect to node i
        self.right(i)

        # If left child exists and is greater than node i
        if self.l < self.heapsize and self.A[self.l] > self.A[i]:
            largest = self.l # left child is largest
        else:
            largest = i
        
        # If right child exists and is greater than node i
        if self.r < self.heapsize and self.A[self.r] > self.A[largest]:
            largest = self.r # right child is largest

        if largest != i:
            # Swap node i and largest value
            self.A[i], self.A[largest] = self.A[largest], self.A[i]
            self.heapify(largest) # recursively heapify at largest value
    
    # Build a heap
    def build_heap(self):
 
        for i in range(self.heapsize // 2 - 1, -1, -1):
            self.heapify(i) # recursively heapify at node i
    
    # Execute heapsort (directly modifies A attribute)
    def heapsort(self):
        # Invoke build_heap method
        self.build_heap()

        for i in range(self.heapsize - 1, 0, -1):
            # Swap root and node i
            self.A[0], self.A[i] = self.A[i], self.A[0]
            self.update_size() # subtract 1 from heapsize
            self.heapify(0) # recursively heapify at root

# Initialize failure count
failures = 0
# Iterate through 100 different random arrays
for j in range(1, 100):
    # Generate random test array
    test_array = np.random.randint(-999, 999, size=2000)
    # Instantiate an arbitrary heap
    test_heap = Heap(test_array)
    # Sort the test heap with heapsort
    test_heap.heapsort()
    
    # Ensure heap sort returns same result as np.sort
    if sum(test_heap.A != np.sort(test_array)):
        print('Random testing failed!')
        failures += 1
if failures == 0:
    print('Passed all random tests!')
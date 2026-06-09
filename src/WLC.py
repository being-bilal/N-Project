# Implementation of the WLC (Weighted Linear Combination) method 
import pandas as pd
import numpy as np


criteria = ['is_protected', 'dist_to_water', 'seismic_pga', 'population', 
            'dist_to_fault', 'dist_to_volcano', 'border_distance']

# Assigning weights based on the AHP method to construct the pairwise comparison matrix
n = len(criteria)
A = np.ones((n, n))
A[0, 1] = 2.0  
A[0, 2] = 3.0  
A[0, 3] = 3.0  
A[0, 4] = 5.0  
A[0, 5] = 5.0  
A[0, 6] = 7.0  
A[1, 2] = 2.0  
A[1, 3] = 2.0  
A[1, 4] = 4.0  
A[1, 5] = 4.0  
A[1, 6] = 6.0  
A[2, 3] = 1.0  
A[2, 4] = 3.0  
A[2, 5] = 3.0
A[2, 6] = 5.0  
A[3, 4] = 3.0  
A[3, 5] = 3.0  
A[3, 6] = 5.0  
A[4, 5] = 1.0  
A[4, 6] = 3.0  
A[5, 6] = 3.0  

# Automatically calculate lower reciprocal 
for i in range(n):
    for j in range(i+1, n):
        A[j, i] = 1.0 / A[i, j]

# Computing weights using the AHP method
column_sums = A.sum(axis=0) # Sum of the values in each column
normalized_A = A / column_sums # Normalize each element by the sum of its column
weights = normalized_A.mean(axis=1) # Average of each row gives the priority vector (weights)

### This is the weight that are be used in the WLC method
print("Weights for each Factor:")
for crit, w in zip(criteria, weights):
    print(f"{crit}: {w:.2f}")
    
"""
Weights for each Factor:
is_protected: 0.34
dist_to_water: 0.22
seismic_pga: 0.14
population: 0.14
dist_to_fault: 0.06
dist_to_volcano: 0.06
border_distance: 0.03
"""

### WLC Calculation



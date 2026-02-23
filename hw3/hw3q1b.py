import numpy as np
from rk import check_order

epsilon= 1e-12


gamma= 1/2 + np.sqrt(3)/6
A= np.array([
    [gamma, 0],
    [1-2*gamma, gamma]
])
b= np.array([1/2, 1/2])

print(f'The method has order {check_order(A,b)}')
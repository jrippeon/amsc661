import numpy as np

epsilon= 1e-12

def check_order(A, b):
    '''
    Check the order of a Runge-Kutta method up to order 3,
    using the conditions.
    '''

    (m,n)= A.shape
    assert m == n and b.shape == (n, ), 'improper dimenensions'

    # 1st order check
    if not np.isclose(np.einsum('l->', b), 1):
        return 0

    # 2nd order check
    if not np.isclose(2 * np.einsum('l,lq->', b, A), 1):
        return 1

    # 3rd order check
    if not (np.isclose(3 * np.einsum('l, lq, lr->', b, A, A), 1) \
        and np.isclose(6 * np.einsum('l, lq, qr->', b, A, A), 1)):
        return 2
    
    return 3


gamma= 1/2 + np.sqrt(3)/6
A= np.array([
    [gamma, 0],
    [1-2*gamma, gamma]
])
b= np.array([1/2, 1/2])

print(f'The method has order {check_order(A,b)}')
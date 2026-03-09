import numpy as np

def verlet(dT, dU, p0, q0, t_span, h):
    '''
    Solve the Hamiltonian system given by H(p,q) = T(p) + U(q).
    dT and dU are the gradient of T and U with respect to their parameters.
    '''
    # Input validation
    
    if not (p0.shape == q0.shape):
        raise ValueError('Incompatible dimensions: p0 and q0 must have the same shape.')

    # Copy initial conditions so we don't accidentally mutate them
    p0= np.array(p0, dtype=float)
    q0= np.array(q0, dtype=float)

    # initialize time, momentum, and position arrays
    t= np.arange(t_span[0], t_span[1], h)
    n_max= len(t)
    p= np.zeros((n_max,) + p0.shape)
    q= np.zeros((n_max,) + q0.shape)
    p[0], q[0]= p0, q0

    for n in range(1, n_max):
        p_half= p[n-1] - h/2 * dU(q[n-1])
        q[n]= q[n-1] + h * dT(p_half)
        p[n]= p_half - h/2 * dU(q[n])
    
    return t, p, q
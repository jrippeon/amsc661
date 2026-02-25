import numpy as np
from rk import dirk, newton

def linear_multistep(func, t_span, y0, h, a, b, b_1=None, dfdy=None):
    '''
    Implement constant order constant timestep linear multistep method with given coefficients.

    u[n+1] + a[0] * u[n] + a[1] * u[n-1] + ... + a[k-1] * u[n-k+1] = h * (b_1 * f[n+1] + b[0] * f[n] + ... + b[k-1] * f[n-k+1])
    '''

    # k is the number of steps
    k,= a.shape
    if b.shape != (k,):
        raise ValueError('Invalid Dimensions. a and b must have the same length')

    # copy our initial condition so we don't mutate it
    y0= np.array(y0, dtype=float)
    if y0.ndim > 1:
        raise ValueError(f'Invalid Dimensions. {y0.shape=}, must be a scalar or 1D array')

    # Initialize the time array
    t= np.arange(t_span[0], t_span[1], h)
    n_max= len(t)

    # initialize u array to hold our values, we use this to handle the scalar and vector cases
    u= np.zeros((n_max,) + y0.shape)
    u[0]= y0
    f= np.zeros_like(u)


    # initialize an identity matrix, this might be a scalar depending
    I = 1.0 if (y0.ndim == 0) else np.eye(y0.shape[0])

    # We need u[0] through u[k-1] initialized to start the method, we'll do this with dirk
    _, u_init= dirk(
        f=func,
        dfdy=dfdy,
        t_span=[t[0], t[k]], # I want to get k steps, this is a really braindead way to do it
        y0= y0,
        method='DIRK2',
        h=h
    )
    u[:k]= u_init[:k]
    
    # we also need to initialize our history of f values
    # I'm too lazy to make sure this will work in a vectorized way, O(k) time in a python loop I guess
    for n in range(k):
        f[n]= func(t[n], u[n])

    # if there is a b[-1] term defined, then we have an implicit method (gross)
    implicit= (b_1 != None)

    # define these outside the loop to save the time allocating them
    # (I convinced myself it made a difference in the DIRK code, maybe it's just placebo optimization)
    if implicit:
        # a[::-1] reverses a, since it is defined backwards of how we need
        F= lambda x, n: x - h * b_1 * func(t[n], x) + u[n-k:n].T @ a[::-1] - h * f[n-k:n].T @ b[::-1]
        DF= lambda x,n: I - h * b_1 * dfdy(t[n], x)

    # note we start on step k
    for n in range(k, n_max):
        if implicit:
            u[n]= newton(
                f=F,
                fprime=DF,
                x0=u[n-1],
                args=(n,)
            )
        else:
            u[n]= h * f[n-k:n].T @ b[::-1] - u[n-k:n].T @ a[::-1]
        f[n]= func(t[n], u[n])
    
    return t, u
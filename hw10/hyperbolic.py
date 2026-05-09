import numpy as np
from scipy.sparse import diags


# def banded(n, values, diagonals):
#     '''
#     Return the (n, n) matrix with values[i] on diagonal diagonals[i]
#     '''
#     result= np.zeros((n,n))
#     for i in range(len(values)):
#         d= diagonals[i]
#         v= values[i]
#         # find the length of this diagonal
#         l= n - np.abs(d)
#         result += np.diag(np.full(l, v), d)
#     return result

def initialize(xmin, xmax, tmax, h, k, u0):
    x= np.arange(xmin, xmax, h)
    M= len(x)

    # we will either get a function as an initial condition or a vector
    if callable(u0):
        u0vec= u0(x)
    else:
        if len(u0) != M:
            raise ValueError(f'u0 vector has length {len(u0)}, expected {len(x)}')
        u0vec= u0

    t= np.arange(0, tmax, k)
    N= len(t)

    u= np.zeros((N, M))
    u[0]= u0vec
    return t, x, u, N, M

# NOTE: all three of these functions assume f(0) = 0, without that it all breaks

def lax_fr_update(u, f, h, k):
    M= u.shape[0]
    # the creation of these matrices should be factored out of this function for
    # performance, I'll do it later
    D1= diags([1,1], [1, -1], (M,M))
    D2= diags([1, -1], [1, -1], (M,M))
    u_next= 0.5 * D1 @ u - k/(2*h) * D2 @ f(u)

    return u_next

def richtmeyer_update(u,f,h,k):
    M= u.shape[0]

    D1= diags([1,1], [0, 1], (M, M))
    D2= diags([-1, 1], [0, 1], (M,M))

    u_half= 1/2 * D1 @ u - k/(2*h) * D2 @ f(u)

    D3= diags([-1, 1], [-1, 0], (M,M))

    u_next= u - k/h * D3 @ f(u_half)

    return u_next

def maccormack_update(u,f,h,k):
    M= u.shape[0]

    D1= diags([1, -1], [1, 0], (M,M))

    u_star= u - k/h * D1 @ f(u)

    D2= diags([1, -1], [0, -1], (M,M))

    u_next= 0.5 * (u + u_star) - k/(2*h) * D2 @ f(u_star)

    return u_next


def numerically_solve(f, update, xmin, xmax, tmax, h, k, u0):
    '''
    Solve u_t + [f(u)]_x = 0
    u(x,0) = u0,
    0 < t < tmax,
    xmin < x < xmax
    using the rule u^{n+1} = update(u^n, f, h, k)

    We assume that u(x,t)=0 for x outside the domain,
    and f(0) = 0
    '''

    t, x, u, N, M= initialize(xmin, xmax, tmax, h, k, u0)

    for n in range(1, N):
        u[n]= update(u[n-1], f, h, k)

    return t, x, u

def godunov(f, fmin, fmax, xmin, xmax, tmax, h, k, u0):
    '''
    Solve:

    u_t + [f(u)]_x = 0
    u(x,0) = u0,
    0 < t < tmax,
    xmin < x < xmax

    using Godunoffs method.
    fmin(uL, uR) and fmax(uL, uR) return the min and max values of u in [uL, uR]
    '''
    t, x, u, N, M= initialize(xmin, xmax, tmax, h, k, u0)


    def F(uL, uR):
        if uL <= uR:
            return fmin(uL, uR)
        else:
            return fmax(uL, uR)

    for n in range(1, N):
        for j in range(M):
            uj= u[n-1, j]
            ujm1= 0 if j-1 < 0 else u[n-1, j-1]
            ujp1= 0 if j+1 >= M else u[n-1, j+1]
            u[n, j] = uj - k/h * (F(uj, ujp1) - F(ujm1, uj))
    return t, x, u
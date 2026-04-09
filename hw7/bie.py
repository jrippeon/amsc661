import numpy as np
import matplotlib.pyplot as plt


def d(x,y,normal):
    '''
    Given x of size (N, 2), y of size (M,2), returns the (N, M) matrix 
    of all the pairwise d(x,y) = n(y) * (x-y) / (2 π |x - y|^2) values.
    Note that of courjo
    se we require a (M,2) array of normal(y) values.
    '''
    # regularize the inputs no they don't immediately crash our code
    x= np.atleast_2d(x)
    y= np.atleast_2d(y)
    normal= np.atleast_2d(normal)

    # when one of the axes is 1, it gets duplicated along the corresponding axis
    # of the other one. 
    # e.g. if we have a == [1,2,3], b == [2,4,6], then
    # a[None, :] * b[:, None] =
    #
    #[[1, 2, 3]    [[2, 2, 2]   [[2,  4,  6],
    # [1, 2, 3]  *  [4, 4, 4]  = [4,  8, 12], 
    # [1, 2, 3]]    [6, 6, 6]]   [6, 12, 18]

    # (N, M, 2) = (N, 1, 2)  -    (1, M, 2)
    x_minus_y= x[:, None, :] - y[None, :, :]

    # (N, M)                      (N, M, 2)
    abs_x_minus_y_squared= np.sum(x_minus_y**2, axis=-1)

    # (N, M)=              (1, M, 2)    * (N, M, 2)
    dot_prod= np.sum(normal[None, :, :] * x_minus_y, axis=-1)

    #(N,M)=  (N, M)  /                 (N, M)
    result= dot_prod / (2 * np.pi * abs_x_minus_y_squared)

    # this apparently gets rid of the extra dimensions, if atleast_2d added them
    return np.squeeze(result)

# We will get our boundary as G : [0,T] --> R^2 (and derivatives), as well
# as normal : [0, T] --> R^2


def sigma_value(G, dG, ddG, normal, f, t):
    '''
    Get the value of σ on the boundary at points specified by t,
    by approximately solving (-0.5 I + D)σ = f.
    Here, D is the double layer operator.
    '''
    N= len(t)

    h= t[1] - t[0]

    x= G(t)
    dx= dG(t)
    ddx= ddG(t)
    n= normal(t)
    speed= np.linalg.norm(dx, axis=-1)

    # D[i,j] = d(x_i, x_j) * |G'(t_j)| * h

    # compute all the pairwise d(x,y)'s, the diagonal will be messed up
    # need to broadcast the speed across the x axis (we care about speed at y)
    D= d(x,x,n) * speed[None, :] * h

    # correct the diagonal with the formula from Remark 12.2 
    d_diag= (dx[:, 1] * ddx[:, 0] - dx[:, 0] * ddx[:, 1]) / (4 * np.pi * speed**3)

    np.fill_diagonal(D, d_diag * speed * h)

    I= np.eye(N)

    A= D - 0.5 * I

    f_vec= f(t)

    sigma= np.linalg.solve(A, f_vec)

    return sigma

def eval_point(x, G, dG, normal, sigma, t):
    '''
    Evaluate the value of u(x), based on a computed σ and parameterization.
    '''

    h= t[1]-t[0]

    y= G(t)
    dy= dG(t)
    n= normal(t)
    speed= np.linalg.norm(dy, axis=-1)

    # x.shape == (N, 2), y.shape == (M, 2), want (N,) array of u(x)
    #         (N, M)  *       (1, M)   *      (1, M)
    summand= d(x,y,n) * sigma[None, :] * speed[None, :] * h
    result= np.sum(summand, axis=-1)

    return result
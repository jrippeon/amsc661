import numpy as np
import matplotlib.pyplot as plt

# We will get our boundary as G : [0,T] --> R^2


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
    speed= np.linalg.norm(dx, axis=-1)

    d_kernel= lambda i,j: np.inner(normal(t[j]), (x[i]-x[j])) / (2 * np.pi * np.linalg.norm(x[i]-x[j])**2)

    # Compute the diagonal values of D using the formula from Remark 12.2
    d_diag= (dx[:, 1] * ddx[:, 0] - dx[:, 0] * ddx[:, 1]) / (4 * np.pi * speed**3)

    D= np.diag(d_diag * speed * h)

    # assign the off-diagonal elements
    for i in range(N):
        for j in range(N):
            if i != j:
                wj= speed[j] * h
                D[i,j]= d_kernel(i,j) * wj

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
    speed= np.linalg.norm(dy, axis=-1)

    # the einsum does inner products of corresponding vectors
    d_kernel= np.einsum('ij,ij->i', normal(t), (x-y)) / (2 * np.pi * np.linalg.norm(x - y, axis=-1)**2)

    return np.sum(d_kernel * sigma * speed) * h

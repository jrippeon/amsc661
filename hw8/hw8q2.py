# %%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# %% Set up the problem
n= 100
h=  2 / (n+1)
### IMPORTANT: I'm going to have xi of length n+2 so the indices line up
###            (also makes the logic slightly cleaner-looking)
xi= np.linspace(-1, 1, n+2)

# discrete first derivative matrix
D1= np.diag(np.ones(n+1), 1) - np.diag(np.ones(n+1), -1)
# # replace the first and last entries with second-order accurate approximations
D1[0,:3]= np.array([-3, 4, -1])
D1[-1, -3:]= np.array([1, -4, 3])
D1 /= (2 * h)

# discrete second derivative matrix
D2= np.diag(-2*np.ones(n+2)) + np.diag(np.ones(n+1), 1) + np.diag(np.ones(n+1), -1)
D2[0, :4]= np.array([2, -5, 4, -1])
D2[-1, -4:]= np.array([-1, 4, -5, 2])
D2 /= h**2

def F(t, w):
    '''
    ∂_t u(ξ_j, t) = 1/x_f(t)^2 
    [ 1/2 * [ (1 + ξ_j) ∂_ξ u(1,t) + (1 - ξ_j) ∂_ξ u(-1, t)] ∂_ξ u(ξ_j, t) 
    + u(ξ_j, t) ∂_ξξ u(ξ_j, t) + (∂_ξ u(ξ_j, t))^2 ]

    w = (u_0, ..., u_{n+1}, x_L, x_R)
    '''
    u= w[:-2]
    xL= w[-2]
    xR= w[-1]
    xf= 1/2 * (xR-xL)
    # spacial derivatives of u
    du= D1 @ u
    ddu= D2 @ u

    dt_u= 1 / xf**2 * (
        -1/2 * ((1 + xi) * du[n+1] + (1 - xi) * du[0]) * du 
        + u * ddu + (du)**2
    )
    # ∂_ξ u(1,t) = du[n+1]
    dt_xR= -du[n+1] / xf
    dt_xL= -du[0] / xf
    result= np.zeros_like(w)
    result[:-2]= dt_u
    result[-2]= dt_xL
    result[-1]= dt_xR

    return result


# %% obtain solutions

u0_exact= 1 - xi**2
w0_exact= np.zeros(n+4)
w0_exact[:n+2]= u0_exact
w0_exact[-2]= -1
w0_exact[-1]= 1

u0_other= 1 - 0.99 * np.cos(2 * np.pi * xi)
w0_other= np.zeros(n+4)
w0_other[:n+2]= u0_other
w0_other[-2]= -1
w0_other[-1]= 1


sol= solve_ivp(F, [0, 1.2], w0_other, method='BDF')

# %% post-processing

t= sol.t
y= sol.y
# first coord is time, second is space
y= y.T
xR= y[:, -2]
xL= y[:, -1]
y= y[:, :-2]

# undo the change of variables to get x(t) from ξ
xo= 1/2 * (xL + xR)
xf= 1/2 * (xL - xR)
x= xi[None, :] * xf[:, None] + xo[:, None]
print(f'{x.shape=}')
print(f'{t.shape=}')
print(f'{y.shape=}')

ax= plt.figure().add_subplot(projection='3d')
ax.plot_surface(x, t[:, None], y)
plt.show()


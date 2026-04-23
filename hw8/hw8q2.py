# %%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#%%% Set up the problem %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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


#%%% obtain solutions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t_eval= np.arange(0, 1.2, 0.1)

def w0(f):
    u0= f(xi)
    w0= np.zeros(n+4)
    w0[:n+2]= u0
    w0[-2]= -1
    w0[-1]= 1

    return w0

w0_exact= w0(lambda x: 1 - x**2)
w0_other= w0(lambda x: 1 - 0.99 * np.cos(2 * np.pi * x))


sol_exact= solve_ivp(F, [0, 1.2], w0_exact, method='BDF', t_eval=t_eval)
sol_other= solve_ivp(F, [0, 1.2], w0_other, method='BDF', t_eval=t_eval)

#%%% plotting functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def post_process(sol):
    t= sol.t
    y= sol.y
    # first coord is time, second is space
    y= y.T
    xR= y[:, -2]
    xL= y[:, -1]
    u= y[:, :-2]

    # undo the change of variables to get x(t) from ξ(t)
    xo= 1/2 * (xL + xR)
    xf= 1/2 * (xL - xR)
    x= xi[None, :] * xf[:, None] + xo[:, None]

    return t, x, u

def plot_figures_normalized(sol, xi):
    n= len(xi) - 2
    t, x, u= post_process(sol)

    # get u(ξ,t) / u_max (t)
    u_max= np.max(u, axis=-1)
    u_normalized= u / u_max[:, None]

    # pretty colors
    n= len(t)
    colors= plt.cm.turbo(np.linspace(0.2, 0.8, n))

    fig, ax= plt.subplots()

    for i in range(len(t)):
        ax.plot(xi,u_normalized[i], color=colors[i])

    ax.plot(xi, 1 - xi**2, color='k', linestyle=':')    
    legend_labels= [f't={ti:.1f}' for ti in t] + [r'$1 - \xi^2$']
    ax.legend(legend_labels, bbox_to_anchor=(1, 1))
    ax.set_xlabel(r'$\xi$')
    ax.set_ylabel(r'$\frac{u(\xi, t)}{u_\text{max}(t)}$')
    return fig, ax

def plot_figures(sol, xi):
    n= len(xi) - 2
    t, x, u= post_process(sol)

    # pretty colors
    n= len(t)
    colors= plt.cm.turbo(np.linspace(0.2, 0.8, n))

    fig, ax= plt.subplots()

    for i in range(len(t)):
        ax.plot(x[i], u[i], color=colors[i])
    legend_labels= [f't={ti:.1f}' for ti in t]
    plt.legend(legend_labels, bbox_to_anchor=(1,1))
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$u(x, t)$')
    return fig, ax



#%%% Create Plots %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

_, ax= plot_figures(sol_exact, xi)
ax.set_title(f'Solution with parabolic IC')
plt.savefig("../../HW8/exact.eps", bbox_inches="tight")
plt.show()

_, ax= plot_figures_normalized(sol_exact, xi)
ax.set_title(f'Normalized solution with parabolic IC')
plt.savefig("../../HW8/exact_normalized.eps", bbox_inches="tight")
plt.show()

_, ax= plot_figures(sol_other, xi)
ax.set_title(f'Solution with sinusoidal IC')
plt.savefig("../../HW8/other.eps", bbox_inches="tight")
plt.show()

_, ax= plot_figures_normalized(sol_other, xi)
ax.set_title(f'Normalized solution with sinusoidal IC')
plt.savefig("../../HW8/other_normalized.eps", bbox_inches="tight")
plt.show()
# %%

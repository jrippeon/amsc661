# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
# %% for plotting
def plot_RAS(
    A,
    b,
    window_size= 3,
    num_points=1000,
    filled=False,
    ax=None,
    color='k',
    label=None
):
    '''
    Plots the region of absolute stability for an RK method with the given
    Butcher's array `A` and `b`.
    '''

    #### actually computing the contour ####

    # we can get the number of stages from the size of A
    s= A.shape[0]
    ones= np.ones(s)
    I= np.eye(s)

    # mesh grid of z values
    x= np.linspace(-window_size, window_size, num_points)
    y= np.linspace(-window_size, window_size, num_points)
    X,Y= np.meshgrid(x,y)

    # here z = h * lambda
    Z= X + (1j * Y)

    # we want to compute R(z) = 1 + z b^T (I - zA)^{-1} 1
    # we flatten Z and broadcast everything
    Z_flat= Z.ravel()
    LHS= I - Z_flat[:, None, None] * A
    # this inverts LHS
    sol= np.linalg.solve(LHS, ones)
    # we need to sol @ b to preserve the shapes
    R_flat= 1 + Z_flat * (sol @ b)
    R= R_flat.reshape(Z.shape)
    magnitude= np.abs(R)

    #### plotting ####
    assert ax != None, 'pass axes idiot'

    if filled: # this is easier to see what is inside vs outside, for testing
        ax.contourf(X,Y,magnitude, levels=[0, 1], colors='r')
    else:
        ax.contour(X,Y,magnitude, levels=[1], colors=color)

        # doing this nonsense to have something show up in the legend
        # too lazy to figure out how to do it the right way
        ax.plot([], [], color=color, label=label)

# %% Butchers arrays for various RK methods

# lol 
A_euler= np.array([
    [0]
])
b_euler= np.array([1])

# midpoint with Euler Predictor
A_mep= np.array([
    [0, 0],
    [1/2, 0]
])
b_mep= np.array([0, 1])

A_kutta= np.array([
    [0, 0, 0],
    [1/2, 0, 0],
    [-1, 2, 0]
])
b_kutta= np.array([1/6, 2/3, 1/6])

A_44= np.array([
    [0, 0, 0, 0],
    [1/2, 0, 0, 0],
    [0, 1/2, 0, 0],
    [0, 0, 1, 0]
])
b_44= np.array([
    1/6, 1/3, 1/3, 1/6
])

A_dopri= np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [1/5, 0, 0, 0, 0, 0, 0],
    [3/40, 9/40, 0, 0, 0, 0, 0],
    [44/45, -56/15, 32/9, 0, 0, 0, 0],
    [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
    [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
])
b_dopri= np.array([
    35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0
])


# %% plot
window_size= 4

colors=colormaps['viridis'](np.linspace(0, 0.9, 5))

fig, ax= plt.subplots()
plot_RAS(
    A=A_euler,
    b=b_euler,
    window_size=window_size,
    ax=ax,
    color=colors[0],
    label='Euler'
)
plot_RAS(
    A=A_mep,
    b=b_mep,
    window_size=window_size,
    ax=ax,
    color=colors[1],
    label='Midpoint with Euler Predictor'
)
plot_RAS(
    A=A_kutta,
    b=b_kutta,
    window_size=window_size,
    ax=ax,
    color=colors[2],
    label='Kutta\'s Method'
)
plot_RAS(
    A=A_44,
    b=b_44,
    window_size=window_size,
    ax=ax,
    color=colors[3],
    label='4-stage 4th-order RK'
)
plot_RAS(
    A=A_dopri,
    b=b_dopri,
    window_size=window_size,
    ax=ax,
    color=colors[4],
    label='DOPRI5'
)

ax.vlines([0], -window_size, window_size, colors='k', linestyles='dashed')
ax.hlines([0], -window_size, window_size, colors='k', linestyles='dashed')
ax.set_aspect('equal')
ax.set(
    xlabel='$\\mathrm{Re}(h \\lambda)$',
    ylabel='$\\mathrm{Im}(h \\lambda)$',
    title='RAS for Several Runge-Kutta Methods'
)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()
# %%

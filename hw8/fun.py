#%%% Imports %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import matplotlib.pyplot as plt
import fem_nonlinear as fem
from importlib import reload
reload(fem)

#%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
spam_plots= False
dt= 0.05
t_max= 10

#%%% Load the mesh %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pts= np.loadtxt('pts.csv')
tri= np.loadtxt('tri.csv', dtype=int)
boundary= np.loadtxt('boundary.csv', dtype=int)
Npts= pts.shape[0]
Ntri= tri.shape[0]

if spam_plots:
    plt.triplot(pts[:,0], pts[:,1], tri)
    plt.scatter(pts[boundary, 0], pts[boundary, 1], c='r')
    plt.show()
#%%% Boundary conditions and heating %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def f(x, t):
    '''
    Takes x of shape (Npts, 2) and t of shape (N), returns values of f(x,t) 
    at each point and timestep of shape (N, Npts)
    '''
    Npts= x.shape[0]
    N= t.shape[0]
    result= np.zeros((N, Npts))

    # do nothing, no body heating

    return result
def g(x,t):
    '''
    Takes x of shape (Npts, 2) and t of shape (N), returns values of g(x,t) 
    at each point and timestep of shape (N, Npts)
    '''
    Npts= x.shape[0]
    N= t.shape[0]

    # shape (Npts,)
    r= np.linalg.norm(x, axis=-1)
    phi= np.atan2(x[:, 1], x[:,0])

    # spinning heat on boundary 
    omega= 4

    # result= np.cos(phi[None, :] + omega * t[:, None]) 
    result= np.cos(phi[None, :] + omega * t[:, None]) * (r < 1.5)[None,:] \
          + np.cos(phi[None, :] - omega * t[:, None]) * (r > 1.5)[None,:]
    return result

def a_evil(x):
    phi= np.atan2(x[:, 1], x[:,0])
    r= np.linalg.norm(x, axis=-1)
    return np.cos(3 * phi) + 0.99 # go just a tiny bit negative >:)

#%%% Visualize the boundary conditions we're setting %%%%%%%%%%%%%%%%%%%%%%%%%%%
if spam_plots:
    t= np.arange(0, t_max, dt)
    N= t.shape[0]
    u= g(pts,t)
    plt.tricontourf(pts[:,0], pts[:,1], tri, u[30], levels=200, cmap='magma')

    ani= fem.animate_solution(pts, tri, u, t)
    plt.show()
#%% Initial condition %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# u0= np.zeros((Npts,))
u0= g(pts, np.array([0]))[0]

#%%% Caluculate the response with given boundary conditions %%%%%%%%%%%%%%%%%%%%
reload(fem)
t, u= fem.heat(
    pts=pts,
    tri=tri,
    boundary=boundary,
    u0=u0,
    dt=dt,
    t_max=t_max,
    a=1,
    g=g,
    time_dependent=True
)
# ani= fem.animate_solution(pts, tri, np.log(u), t)
ani= fem.animate_solution(pts, tri, u, t)
plt.show()
ani.save('fun.mp4', fps=30)
# %%

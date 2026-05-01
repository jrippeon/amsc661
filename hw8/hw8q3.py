# %%
import fem_nonlinear as fem
import numpy as np
import matplotlib.pyplot as plt
import distmesh as dm
from importlib import reload
from matplotlib.animation import FuncAnimation
from time import time

#%%% create the mesh %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def f_unif_ones(p):
    p= np.array(p)
    n, _= p.shape
    return np.ones(n)

def f_unif_zeros(p):
    p= np.array(p)
    n, _= p.shape
    return np.zeros(n)

# h= 0.05

# sdf= dm.ddiff(
#     dm.dcircle(0, 0, 2),
#     dm.dcircle(0, 0, 1)
# )


# theta= np.linspace(0, 2 * np.pi, 10)
# pfix= np.array([np.sin(theta), np.cos(theta)]).T

# pts, tri= dm.distmesh2D(
#     fd=sdf,
#     fh=f_unif_ones,
#     h0=h,
#     pfix=pfix,
#     bbox=(-2,2,-2,2)
# )
#%%% Save / Load mesh %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# np.savetxt('pts.csv', pts)
# np.savetxt('tri.csv', tri, fmt='%d')
pts= np.loadtxt('pts.csv')
tri= np.loadtxt('tri.csv', dtype=int)

#%%% locate boundary nodes %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
boundary_tol= 1e-6
r= np.linalg.norm(pts, axis=-1)
inner_boundary= np.nonzero(r < 1 + boundary_tol)
outer_boundary= np.nonzero(r > 2 - boundary_tol)
boundary= np.union1d(inner_boundary, outer_boundary)

#%%% visualize mesh %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# plt.triplot(pts[:,0], pts[:,1], tri)
# plt.scatter(pts[boundary, 0], pts[boundary, 1], c='r')
# plt.show()

#%%% initial condition %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
r= np.linalg.norm(pts, axis=-1)
phi= np.atan2(pts[:,1], pts[:,0])

u0= r + np.cos(phi)

# #%%% Wacky BC (just for fun) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# u0[inner_boundary]= 10 * np.cos(phi[inner_boundary])
# u0[outer_boundary]= 10 * np.sin(phi[outer_boundary])
# uD= u0[boundary]

#%%% Normal BC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u0[boundary]= 0

#%%% run the code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
reload(fem)
dt= 0.01
t_max= 1

# we can reuse the f from above here, just needs to return 1 lol
f= f_unif_ones

t, u= fem.heat(
    pts=pts,
    tri=tri,
    boundary=boundary,
    f=f,
    u0=u0,
    dt=dt,
    t_max=t_max
)
#%%% For plotting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_surface(pts, tri, t, u, t_plot):
    '''
    Plot a surface of u at time t_plot
    '''

    # find the index of the nearest time
    i_plot= np.abs(t - t_plot).argmin()

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.plot_trisurf(pts[:, 0], pts[:, 1], u[i_plot], cmap='magma')
    ax.set_title(f'$u$ vs $(x,y)$ at $t={t_plot}$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    return fig, ax
def plot_radial(pts, tri, t, u, t_plot):
    # find the index of the nearest time
    i_plot= np.abs(t - t_plot).argmin()

    r= np.linalg.norm(pts, axis=-1)
    r_sorted_indices= np.argsort(r)
    fig, ax= plt.subplots()
    ax.scatter(r[r_sorted_indices], u[i_plot, r_sorted_indices], s=3, c=u[i_plot, r_sorted_indices], cmap='magma')
    ax.set_title(f'$u$ vs $r$ at $t={t_plot}$')
    ax.set_xlabel('r')
    ax.set_ylabel('u')
    return fig, ax
#%%% Generate plots %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig, ax= plot_surface(pts, tri, t, u, 0.1)
plt.show()
fig, ax= plot_surface(pts, tri, t, u, 1)
plt.show()

# radial graph
fig, ax= plot_radial(pts, tri, t, u, 1.1)
# plot exact solution
r= np.linspace(1, 2, 100)
u_exact= (1 - r**2)/4 + 3 * np.log(r) / (4 * np.log(2))
ax.plot(r, u_exact, color='k', linestyle=':')
plt.show()
# %%


# Nonsense for testing
# # %%
# fig, ax= plt.subplots()
# ax.set_aspect('equal')
# mesh= ax.tripcolor(pts[:,0], pts[:,1], tri, u[50], shading='gouraud', cmap='magma')
# mesh.set_array(u[50])
# fig.colorbar(mesh, ax=ax)

# # %% cute animation 
# #(AI CODE WARNING, TOO LAZY TO FIGURE THIS OUT ON MY OWN FOR A RANDOM VISUALIZATION)
# fig, ax= plt.subplots()
# ax.set_aspect('equal')

# mesh= ax.tripcolor(pts[:,0], pts[:,1], tri, u[0], shading='gouraud', cmap='magma')
# fig.colorbar(mesh, ax=ax)
# mesh.set_clim(vmin=u.min(), vmax=u.max())

# def update(frame):
#     mesh.set_array(u[frame])
#     ax.set_title(f't = {t[frame]:.2f}')
#     return mesh, # this comma is apparently crucial, we need this to return an iterable (tuple)

# ani= FuncAnimation(fig, update, frames=u.shape[0], interval=50)
# plt.show()

# # %%
# ani.save('sol.mp4', fps=30)

# %%
import fem_nonlinear as fem
import numpy as np
import matplotlib.pyplot as plt
import distmesh as dm
from importlib import reload
from matplotlib.animation import FuncAnimation

# %% create the mesh

h= 0.1

sdf= dm.ddiff(
    dm.dcircle(0, 0, 2),
    dm.dcircle(0, 0, 1)
)

def f(p):
    p= np.array(p)
    n, _= p.shape
    return np.ones(n)

theta= np.linspace(0, 2 * np.pi, 10)
pfix= np.array([np.sin(theta), np.cos(theta)]).T

pts, tri= dm.distmesh2D(
    fd=sdf,
    fh=f,
    h0=h,
    pfix=pfix,
    bbox=(-2,2,-2,2)
)

# %% locate boundary nodes
boundary_tol= 1e-6
r= np.linalg.norm(pts, axis=-1)
inner_boundary= np.nonzero(r < 1 + boundary_tol)
outer_boundary= np.nonzero(r > 2 - boundary_tol)
boundary= np.union1d(inner_boundary, outer_boundary)

# %% visualize mesh
# plt.triplot(pts[:,0], pts[:,1], tri)
# plt.scatter(pts[boundary, 0], pts[boundary, 1], c='r')
# plt.show()


# %% initial condition
phi= np.atan2(pts[:,1], pts[:,0])

u0= r + np.cos(phi)

# %% run the code
reload(fem)
dt= 0.01
t_max= 1

# we can reuse the f from above here, just needs to return 1 lol

t, u= fem.heat(
    pts=pts,
    tri=tri,
    boundary=boundary,
    f=f,
    u0=u0,
    dt=dt,
    t_max=t_max
)
# %%
fig, ax= plt.subplots()
ax.set_aspect('equal')
mesh= ax.tripcolor(pts[:,0], pts[:,1], tri, u[50], shading='gouraud')
mesh.set_array(u[50])
fig.colorbar(mesh, ax=ax)
# %% cute animation 
#(AI CODE WARNING, TOO LAZY TO FIGURE THIS OUT ON MY OWN FOR A RANDOM VISUALIZATION)
fig, ax= plt.subplots()
ax.set_aspect('equal')
print(f'{u.shape=}')

mesh= ax.tripcolor(pts[:,0], pts[:,1], tri, u[0], shading='gouraud')
fig.colorbar(mesh, ax=ax)
mesh.set_clim(vmin=u.min(), vmax=u.max())

def update(frame):
    mesh.set_array(u[frame])
    ax.set_title(f't = {t[frame]:.2f}')
    return mesh, # this comma is apparently crucial, we need this to return an iterable (tuple)

ani= FuncAnimation(fig, update, frames=u.shape[0], interval=50)
plt.show()

# %%
ani.save('sol.mp4', fps=30)

# %%

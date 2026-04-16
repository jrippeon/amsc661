#%%
import numpy as np
import distmesh as dm
import fem_nonlinear as fem
import matplotlib.pyplot as plt

#%% Create our domain.

h= 0.04

def fh(p):
    p= np.array(p)
    n, _= p.shape
    return np.ones(n)

# create SDF for the square
corners=np.array([
    [-1,-1],
    [1,-1],
    [1,1],
    [-1,1]
])

sdf= dm.dconvex(corners)

# create the mesh 
pts, tri= dm.distmesh2D(
    fd=sdf,
    fh=fh,
    h0=h,
    pfix=corners,
    bbox=(-1,1,-1,1)
)


#%% save / load the mesh
# np.savetxt('pts.csv', pts)
# np.savetxt('tri.csv', tri, fmt='%d')
pts= np.loadtxt('pts.csv')
tri= np.loadtxt('tri.csv', dtype=int)

#%% find the boundary points
bdy_tol= 1e-6
N_idx= np.nonzero(pts[:,0] > 1 - bdy_tol)
S_idx= np.nonzero(pts[:,0] < -1 + bdy_tol)
E_idx= np.nonzero(pts[:,1] > 1 - bdy_tol)
W_idx= np.nonzero(pts[:,1] < -1 + bdy_tol)
boundary= np.union1d(N_idx,np.union1d(S_idx, np.union1d(E_idx, W_idx)))

# %% plot mesh
# plt.scatter(pts[:,0], pts[:,1], s=1)
plt.triplot(pts[:,0], pts[:,1], tri)
plt.scatter(pts[boundary,0], pts[boundary,1], c='r', s=5)
plt.gca().set_aspect('equal')

# %% computing solution with homogeneous dirichlet conditions
Npts= pts.shape[0]

u_init= -np.ones(Npts)
u_init[boundary]= 0
u= fem.ginzburg_landau(pts, tri, boundary, u_init=u_init)

ax= plt.figure().add_subplot(projection='3d')
ax.plot_trisurf(pts[:,0], pts[:,1], tri, u)

# %% computing solution with frustrated dirichlet conditions
Npts= pts.shape[0]
u_init= -np.ones(Npts)
u_init[N_idx]= -1
u_init[S_idx]= -1
u_init[E_idx]= 1
u_init[W_idx]= 1
uD= u_init[boundary]
u_frustrated= fem.ginzburg_landau(pts, tri, boundary, uD=uD)

ax= plt.figure().add_subplot(projection='3d')
ax.plot_trisurf(pts[:,0], pts[:,1], tri, u_frustrated)
# %%

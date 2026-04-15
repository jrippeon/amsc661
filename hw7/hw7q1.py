#%%
import numpy as np
import distmesh as dm
import fem_nonlinear as fem
import matplotlib.pyplot as plt

#%% Create our domain.

h= 0.1

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

# find the boundary points

bdy_tol= 1e-6
N_idx= np.nonzero(pts[:,0] > 1 - bdy_tol)
S_idx= np.nonzero(pts[:,0] < -1 + bdy_tol)
E_idx= np.nonzero(pts[:,1] > 1 - bdy_tol)
W_idx= np.nonzero(pts[:,1] < -1 + bdy_tol)
boundary= np.union1d(N_idx,np.union1d(S_idx, np.union1d(E_idx, W_idx)))

# %% 
plt.scatter(pts[:,0], pts[:,1])
plt.triplot(pts[:,0], pts[:,1], tri)
plt.scatter(pts[boundary,0], pts[boundary,1], c='r')

# %%

u= fem.ginzburg_landau(pts, tri, boundary)


print(u)
# %%
plt.tricontourf(pts[:,0], pts[:,1], tri, u)

# %%

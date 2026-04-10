#%%
import distmesh as dm
import numpy as np
import matplotlib.pyplot as plt
import fem
from importlib import reload
reload(fem)
reload(dm)

#%%#############################################################################
# Generate the mesh
################################################################################

h= 0.1

def fh(p):
    p= np.array(p)
    n, _= p.shape
    return np.ones(n)

# create SDF for the square

sdf= dm.dconvex(np.array([
    [0, 0],
    [3, 0],
    [3, 3],
    [0, 3]
]))

# fix points for the circle (space approximately h apart)

theta= np.arange(0, 2 * np.pi, h)

circle_bdy= np.array([
    np.cos(theta),
    np.sin(theta)
])
circle_bdy= circle_bdy.T
offset= np.array([1.5, 1.5])
circle_bdy += offset

# create the mesh (with circle boundary points remaining fixed)

pts, tri= dm.distmesh2D(
    fd=sdf,
    fh=fh,
    h0=h,
    pfix=circle_bdy,
    bbox=(0, 3, 0, 3)
)


# find the Dirichlet bdy points
tol= 1e-6
# find the indices on the left edge of the square
D0_idx= np.nonzero(pts[:,0] < tol)
D1_idx= np.nonzero(pts[:,0] > 3 - tol)

#%%#############################################################################
# Load a way better mesh I precomputed
################################################################################
pts= np.loadtxt('pts.csv')
tri= np.loadtxt('tri.csv', dtype=np.int64)

# find the Dirichlet bdy points
tol= 1e-6
# find the indices on the left edge of the square
D0_idx= np.nonzero(pts[:,0] < tol)
D1_idx= np.nonzero(pts[:,0] > 3 - tol)

circle_bdy_indices= np.isclose(np.linalg.norm(pts - offset, axis= 1), 1, atol=tol)
circle_bdy= pts[circle_bdy_indices]

#%%#############################################################################
# visualize the mesh created (fixed points in red, boundry points in black)
################################################################################

dot_size=1

dm.visualize_result(pts, tri)
plt.scatter(pts[:,0], pts[:,1], s=0.5*dot_size)
plt.scatter(circle_bdy[:,0], circle_bdy[:,1], c='r', s=dot_size)
plt.scatter(pts[D0_idx, 0], pts[D0_idx, 1], c='k', s=dot_size)
plt.scatter(pts[D1_idx, 0], pts[D1_idx, 1], c='k', s=dot_size)
plt.show()

#%%#############################################################################
# Compute the solution
################################################################################

a_inner= 1.2
a_outer= 1

dirichlet_bdy_segments=[
    (D0_idx, 0),
    (D1_idx, 1),
]

def a(x):
    tol= 1e-7
    # absolute disgusting code
    c= np.array([1.5, 1.5])
    if x.ndim == 1:
        if np.linalg.norm(x - c) <= 1:
            return a_inner
        else:
            return a_outer
    else:
        dist= np.linalg.norm(x - c, axis=1)
        result= a_outer * np.ones_like(dist)
        result[dist <= 1 + tol]= a_inner
        return result


u= fem.laplace_advanced(pts, tri, a, dirichlet_bdy_segments)
#%%#############################################################################
# Plot the voltage
################################################################################
plt.tricontourf(pts[:,0], pts[:,1], tri, u, 100)
# plt.triplot(pts[:,0],pts[:,1],tri,linewidth=0.5, c='k', alpha=0.3)
plt.gca().set_aspect('equal')
plt.title(f'Voltage $v$ with $a_1 = {a_inner}$ and $a_2 = {a_outer}$')
plt.colorbar()
plt.show()


#%%#############################################################################
# Plot the gradient
################################################################################
du= fem.grad(pts, tri, u, faces=True)
plt.tripcolor(pts[:,0], pts[:,1], tri, np.linalg.norm(du, axis=1))
plt.show()
#%%#############################################################################
# Plot the current density calculated properly
################################################################################
j_vertices= fem.flux_density(pts, tri, u, a)

plt.tricontourf(pts[:,0], pts[:,1], tri, j_vertices, 250, cmap='viridis')
# plt.triplot(pts[:,0],pts[:,1],tri,linewidth=0.5, c='k', alpha=0.3)
plt.gca().set_aspect('equal')
plt.title(f'Current density $j$ with $a_1 = {a_inner}$ and $a_2 = {a_outer}$')
plt.colorbar()
plt.show()
#%%#############################################################################
# Plot the current density incorrectly (my first attempt)
################################################################################
du= fem.grad(pts, tri, u)
a_matrix= a(pts)
j= a_matrix * np.linalg.norm(du, axis=1)
plt.tricontourf(pts[:,0], pts[:,1], tri, j, 250, cmap='viridis')
# plt.tripcolor(pts[:,0], pts[:,1], tri, j, shading='gouraud', cmap='viridis')
plt.colorbar()
plt.show()

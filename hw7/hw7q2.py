#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import bie
from importlib import reload
reload(bie)
from scipy.interpolate import CubicSpline

#%%############################################################################# 
# Generate Mesh (lifted directly from Dr. Cameron's code)
################################################################################

# curve r(t) = r1 + r2*cos(m*t)
r1 = 2
r2 = 0.5
m_star = 5

# Define the curve
def star_curve_rad(t):
    return r1 + r2 * np.cos(m_star*t)

def star_curve_drad(t):
    return -r2*m_star*np.sin(m_star*t)
    
def star_curve_speed(t):
    r = star_curve_rad(t)
    dr = star_curve_drad(t)
    return np.sqrt(r**2 + dr**2)

def star_curve_normal(t):
    r = star_curve_rad(t)
    dr = star_curve_drad(t)
    speed = np.sqrt(r**2 + dr**2)
    v1 = np.array([np.cos(t),np.sin(t)])
    v2 = np.array([np.sin(t),-np.cos(t)])
    normal = (r*v1 + dr*v2)/speed
    return normal

def get_point(t):
    r = star_curve_rad(t)
    return np.array([r*np.cos(t),r*np.sin(t)])

def exact_sol(r,t):
    return 1 + (r**3)*np.cos(3*t)
    
def bdry_func(t):
    r = star_curve_rad(t)
    return exact_sol(r,t)

import pygmsh
import numpy as np

h_mesh = 0.1

N = 200
t_vals = np.linspace(0, 2*np.pi, N, endpoint=False)

# Convert to Cartesian
points = np.array([
    [star_curve_rad(t)*np.cos(t), star_curve_rad(t)*np.sin(t), 0.0]
    for t in t_vals
])

with pygmsh.geo.Geometry() as geom:
    # Add points
    pts = [geom.add_point(p, mesh_size=h_mesh) for p in points]

        # Close the loop
    pts.append(pts[0])

    # Create spline
    curve = geom.add_spline(pts)

    # Create surface
    loop = geom.add_curve_loop([curve])
    surface = geom.add_plane_surface(loop)

    # Generate mesh
    mesh = geom.generate_mesh()

# mesh.points → coordinates
# mesh.cells_dict["triangle"] → connectivity
tri = mesh.cells_dict["triangle"]
pts = mesh.points[:, :2]  # ignore z

# plt.figure()
# plt.triplot(pts[:,0], pts[:,1], tri,linewidth = 0.5)
# # plt.scatter(pts[Bdry,0],pts[Bdry,1],c = 'red')
# plt.gca().set_aspect('equal')
# plt.title("Triangulation of the star-shaped region")

# Find boundary nodes and interior nodes
Npts = np.size(pts,axis = 0)
Ntri = np.size(tri,axis = 0)
print(f"Npts = {Npts}, Ntri = {Ntri}")

Nedges = Ntri*3
edges = np.zeros((Nedges,2),dtype = int)
for j in range(Ntri):
    t = tri[j,:]
    edges[3*j,:] = np.sort(np.array([t[0],t[1]]))
    edges[3*j+1,:] = np.sort(np.array([t[0],t[2]]))
    edges[3*j+2,:] = np.sort(np.array([t[1],t[2]]))

# Indices of boundary points
unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
ind = np.argwhere(counts == 1)
Bdry = np.unique(unique_edges[ind,:].ravel())

# Indices of interior points
# Note that pts contains not only mesh points 
# but also points on the boundary curve that did not become part of the mesh
mesh_pts = np.unique(unique_edges.ravel())
Interior = np.setdiff1d(mesh_pts,Bdry)

# plt.figure()
# plt.triplot(pts[:,0], pts[:,1], tri,linewidth = 0.5)
# plt.scatter(pts[Bdry,0],pts[Bdry,1],s = 2, c = 'red')
# # plt.scatter(pts[:,0],pts[:,1],s = 0.5, c = 'blue')
# plt.scatter(pts[Interior,0],pts[Interior,1],s = 1, c = 'black')
# plt.gca().set_aspect('equal')
# plt.show()

#%%#############################################################################
# Find the boundary points (my way)
################################################################################

# Dr. Cameron's code above misses a bunch of the boundary points, I'm going
# to attempt to find them using the radius
phi_values= np.atan2(pts[:,1], pts[:,0])
r_values= np.linalg.norm(pts, axis=-1)
r_exact= star_curve_rad(phi_values)
bdy_tol= 1e-1

# assume a point is in the boundary if it's less than h/10 from the boundary
Bdry= (np.abs(r_exact - r_values) < h_mesh * bdy_tol)


#%%#############################################################################
# Parameterization of the boundary.
################################################################################

def star_curve_ddrad(t):
    return -r2 * m_star**2 * np.cos(m_star*t)

# I redefine these functions to be a bit more vectorized
def G(t):
    r= star_curve_rad(t)
    x0= r * np.cos(t)
    x1= r * np.sin(t)

    return np.array([x0, x1]).T

def dG(t):
    r= star_curve_rad(t)
    dr= star_curve_drad(t)
    dx0= - r * np.sin(t) + dr * np.cos(t)
    dx1= r * np.cos(t) + dr * np.sin(t)

    return np.array([dx0, dx1]).T

def ddG(t):
    r= star_curve_rad(t)
    dr= star_curve_drad(t)
    ddr= star_curve_ddrad(t)

    ddx0= (ddr - r) * np.cos(t) - 2 * dr * np.sin(t)
    ddx1= (ddr - r) * np.sin(t) + 2 * dr * np.cos(t)
    return np.array([ddx0, ddx1]).T

def normal(t):
    speed= np.linalg.norm(dG(t), axis=-1)
    r= star_curve_rad(t)
    dr= star_curve_drad(t)
    n0= (r * np.cos(t) + dr * np.sin(t)) / speed
    n1= (r * np.sin(t) - dr * np.cos(t)) / speed
    return np.array([n0, n1]).T

def f(t):
    return 1 + (r1 + r2 * np.cos(m_star * t))**3 * np.cos(3 * t)

N_bdy_points= 200

t= np.linspace(0, 2 * np.pi, N_bdy_points + 1)
t= np.delete(t, -1) # we double-counted 2pi = 0 (mod 2pi) lol

x= G(t)

# plt.scatter(x[:,0], x[:,1])
# plt.show()
#%%#############################################################################

sigma= bie.sigma_value(G, dG, ddG, normal, f, t)
print(f'{sigma.shape=}')

#%%#############################################################################
# x= pts
# u= bie.eval_point(
#     x=x,
#     G=G,
#     dG=dG,
#     normal=normal,
#     sigma=sigma,
#     t=t
# )

u= np.zeros(Npts)

for i,x in enumerate(pts):
    u[i]= bie.eval_point(
        x=x,
        G=G,
        dG=dG,
        normal=normal,
        sigma=sigma,
        t=t
    )

# # fix the boundary
# for j in range(np.size(Bdry)):
#     ind= Bdry[j]
#     x= pts[ind]
#     phi= np.arctan2(x[1], x[0])
#     u[ind]= bdry_func(phi)

phi= np.atan2(pts[Bdry, 1], pts[Bdry, 0])
r= star_curve_rad(phi)
u[Bdry]= exact_sol(r, phi)
#%%#############################################################################

plt.tricontourf(pts[:,0], pts[:,1], tri, u, levels=200, cmap='turbo')
plt.colorbar()
plt.show()

#%%#############################################################################
# ERROR
################################################################################

u_exact= np.zeros((Npts,))
for j in range(Npts):
    x= pts[j]
    phi= np.arctan2(x[1], x[0])
    r= np.linalg.norm(x)
    u_exact[j]= exact_sol(r, phi)


error= u_exact - u

#%%#############################################################################

plt.tricontourf(pts[:,0], pts[:,1], tri, error, levels=200, cmap='turbo')
plt.colorbar()
plt.show()

#%%#############################################################################
# Interpolate σ to more points
################################################################################

# we need to add one more point to use the interpolation with periodic BC
t_aux= np.zeros((N_bdy_points + 1,))
sigma_aux= np.zeros((N_bdy_points + 1,))
t_aux[:N_bdy_points]= t
t_aux[-1]= 2 * np.pi
sigma_aux[:N_bdy_points]= sigma
sigma_aux[-1]= sigma_aux[0]

# magic stuff from scipy
cs= CubicSpline(t_aux, sigma_aux, bc_type='periodic')

# create a finer mesh
t_fine= np.linspace(0, 2*np.pi, 10 * N_bdy_points + 1)
sigma_fine= cs(t_fine)
t_fine= np.delete(t_fine, -1)
sigma_fine= np.delete(sigma_fine, -1)
N_bdy_points_fine= np.size(t_fine)
#%%#############################################################################
u_fine= np.zeros(Npts)

for i,x in enumerate(pts):
    u_fine[i]= bie.eval_point(
        x=x,
        G=G,
        dG=dG,
        normal=normal,
        sigma=sigma_fine,
        t=t_fine
    )

# # fix the boundary
# for j in range(np.size(Bdry)):
#     ind= Bdry[j]
#     x= pts[ind]
#     phi= np.arctan2(x[1], x[0])
#     u_fine[ind]= bdry_func(phi)

phi_bdry= np.atan2(pts[Bdry, 1], pts[Bdry, 0])
r_bdry= np.linalg.norm(pts[Bdry], axis=-1)
u_fine[Bdry]= exact_sol(r_bdry, phi_bdry)

# this way DOES NOT WORK, the "boundary" points are actually slightly in from the
# mathematical boundary, so you can't evaluate on the boundary, you have to evaluate where they are
# phi_bdry= np.atan2(pts[Bdry, 1], pts[Bdry, 0])
# u_fine[Bdry]= bdry_func(phi_bdry)


#%%#############################################################################
error_fine= u_exact - u_fine
tol= 1e-4
problems= np.abs(error_fine) > tol

plt.tricontourf(pts[:,0], pts[:,1], tri, error_fine, levels=200, cmap='coolwarm')
plt.colorbar()
# plt.scatter(pts[Bdry, 0], pts[Bdry, 1], c='b', s=50, zorder=995)
plt.scatter(pts[problems, 0], pts[problems, 1], c='r', s=100, alpha=0.5, zorder=990)
plt.scatter(pts[:, 0], pts[:, 1], c='k', s=5, zorder=999)
plt.show()
# %%
plt.plot(t_fine, sigma_fine)

# %%

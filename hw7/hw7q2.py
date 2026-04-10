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

#%%#############################################################################
# Find the boundary points (my way)
################################################################################

# Dr. Cameron's code above misses a bunch of the boundary points, I'm going
# to attempt to find them using the radius
phi_values= np.atan2(pts[:,1], pts[:,0])
r_values= np.linalg.norm(pts, axis=-1)
r_exact= star_curve_rad(phi_values)
bdy_tol= 1e-1

# assume a point is in the boundary if it's less than h * tol from the boundary
Bdry= (np.abs(r_exact - r_values) < h_mesh * bdy_tol)


#%%#############################################################################
# Parameterization of the boundary.
################################################################################

# because I did things a funny way you need a bunch of extra nonsense

def star_curve_ddrad(t):
    return -r2 * m_star**2 * np.cos(m_star*t)

# I redefine these functions to be a bit more vectorized
def G(t):
    r= star_curve_rad(t)
    x= r * np.exp(1j * t)
    return np.array([x.real, x.imag]).T

def dG(t):
    r= star_curve_rad(t)
    dr= star_curve_drad(t)
    x= (dr + r * 1j) * np.exp(1j * t)
    return np.array([x.real, x.imag]).T

def ddG(t):
    r= star_curve_rad(t)
    dr= star_curve_drad(t)
    ddr= star_curve_ddrad(t)
    x= (ddr + 2j * dr - r) * np.exp(1j * t)
    return np.array([x.real, x.imag]).T
    
def normal(t):
    dGval= dG(t)
    dx= dGval[:,0] + 1j * dGval[:,1]
    x= -1j * dx / np.abs(dx)
    result= np.array([x.real, x.imag]).T
    return result

def f(t):
    return 1 + (r1 + r2 * np.cos(m_star * t))**3 * np.cos(3 * t)

N_bdy_points= 200

t= np.linspace(0, 2 * np.pi, N_bdy_points, endpoint=False)
# t= np.delete(t, -1) # we double-counted 2pi = 0 (mod 2pi) lol

x= G(t)

#%%#############################################################################
# Compute σ
################################################################################

sigma= bie.sigma_value(G, dG, ddG, normal, f, t)

#%%#############################################################################
# Plot u
################################################################################
u= bie.eval_point(
        x=pts,
        G=G,
        dG=dG,
        normal=normal,
        sigma=sigma,
        t=t
)

phi_bdry= np.atan2(pts[Bdry, 1], pts[Bdry, 0])
r_bdry= np.linalg.norm(pts[Bdry], axis=-1)
u[Bdry]= exact_sol(r_bdry, phi_bdry)

plt.tricontourf(pts[:,0], pts[:,1], tri, u, levels=200, cmap='magma')
plt.colorbar()
plt.show()

#%%#############################################################################
# Plot error
################################################################################

# compute the exact solution and print the error

phi= np.atan2(pts[:, 1], pts[:, 0])
r= np.linalg.norm(pts, axis=-1)
u_exact= exact_sol(r, phi)

error= u - u_exact

plt.tricontourf(pts[:,0], pts[:,1], tri, error, levels=200, cmap='coolwarm')
plt.colorbar()
plt.show()

#%%#############################################################################
# Interpolate σ to more points
################################################################################
t_fine, sigma_fine= bie.refine_sigma(t, sigma, 10)
N_bdy_points_fine= np.size(t_fine)
#%%#############################################################################
# Plot error again
################################################################################

u_fine= bie.eval_point(
    x=pts,
    G=G,
    dG=dG,
    normal=normal,
    sigma=sigma_fine,
    t=t_fine
)


phi_bdry= np.atan2(pts[Bdry, 1], pts[Bdry, 0])
r_bdry= np.linalg.norm(pts[Bdry], axis=-1)
u_fine[Bdry]= exact_sol(r_bdry, phi_bdry)

# this way DOES NOT WORK, the "boundary" points are actually slightly in from the
# mathematical boundary, so you can't evaluate on the boundary, you have to evaluate where they are
# phi_bdry= np.atan2(pts[Bdry, 1], pts[Bdry, 0])
# u_fine[Bdry]= bdry_func(phi_bdry)

error_fine= u_fine - u_exact

plt.tricontourf(pts[:,0], pts[:,1], tri, error_fine, levels=200, cmap='coolwarm')
plt.colorbar()
plt.show()
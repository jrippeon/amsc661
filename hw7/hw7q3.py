#%%
import numpy as np
import matplotlib.pyplot as plt
import bie
from scipy.interpolate import CubicSpline
import pygmsh


#%%#############################################################################
# Define the domain and problem
################################################################################

r1= 2
r2= 0.5
m_star= 5

def star_curve_rad(t):
    return r1 + r2 * np.cos(m_star*t)

def star_curve_drad(t):
    return -r2*m_star*np.sin(m_star*t)

def star_curve_ddrad(t):
    return -r2 * m_star**2 * np.cos(m_star*t)

# you can define these with complex numbers, it's a bit cute but it works
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
    x= -1j * dx / np.abs(dx) # literall just the normalized velocity and turn clockwise
    return np.array([x.real, x.imag]).T

def f(t):
    return (r1 + r2 * np.cos(m_star * t))**(-3) * np.cos(3 * t) + np.log(r1 + r2 * np.cos(m_star * t))

def exact_sol(r,t):
    return r**-3 * np.cos(3 * t) + np.log(r)

#%%#############################################################################
# Mesh the domain
################################################################################

h_mesh= 1
N= 200
t= np.linspace(0, 2*np.pi, N, endpoint=False)
bdy_pts= G(t)
# bdy_pts= np.concat((bdy_pts, np.zeros((N, 1))), axis=1)

with pygmsh.occ.Geometry() as geom:

    geom.characteristic_length_min = h_mesh
    geom.characteristic_length_max = h_mesh

    star= geom.add_polygon(bdy_pts) 

    bbox= geom.add_rectangle([-4, -4, 0], 8, 8)

    exterior= geom.boolean_difference(bbox, star)

    mesh= geom.generate_mesh()

pts = mesh.points
pts= pts[:,:2]
tri = mesh.cells_dict['triangle']


#%%#############################################################################
# Identify boundary points
################################################################################

phi= np.atan2(pts[:,1], pts[:,0])
r= np.linalg.norm(pts, axis=-1)
r_star= star_curve_rad(phi)
bdy_tol= 1e-2

boundary= (np.abs(r_star - r) < h_mesh * bdy_tol)
#%%#############################################################################
# Evaluate and plot u
################################################################################

sigma= bie.sigma_value(G, dG, ddG, normal, f, t, exterior=True)

u= bie.eval_point(
        x=pts,
        G=G,
        dG=dG,
        normal=normal,
        sigma=sigma,
        t=t,
        exterior=True
)


# this is just decorative post-processing
phi_bdry= np.atan2(pts[boundary, 1], pts[boundary, 0])
r_bdry= np.linalg.norm(pts[boundary], axis=-1)
u[boundary]= exact_sol(r_bdry, phi_bdry)

plt.tricontourf(pts[:,0], pts[:,1], tri, u, levels=200, cmap='magma')
# plt.plot(G(t)[:,0], G(t)[:,1], c='k', lw=1)
plt.colorbar()
plt.show()
#%%#############################################################################
# Compute the error
################################################################################

phi= np.atan2(pts[:, 1], pts[:, 0])
r= np.linalg.norm(pts, axis=-1)
u_exact= exact_sol(r, phi)

error= u - u_exact

plt.tricontourf(pts[:,0], pts[:,1], tri, error, levels=200, cmap='coolwarm')
# plt.plot(G(t)[:,0], G(t)[:,1], c='k')
plt.colorbar()
plt.show()
#%%#############################################################################
# Refine and compute error again
################################################################################
t_fine, sigma_fine= bie.refine_sigma(t, sigma, 15)
u_fine= bie.eval_point(
        x=pts,
        G=G,
        dG=dG,
        normal=normal,
        sigma=sigma_fine,
        t=t_fine,
        exterior=True
)
# this is just decorative post-processing
u_fine[boundary]= u[boundary]

error_fine= u_fine - u_exact


plt.tricontourf(pts[:,0], pts[:,1], tri, error_fine, levels=200, cmap='coolwarm')
# plt.plot(G(t)[:,0], G(t)[:,1], c='k')
plt.colorbar()
plt.show()
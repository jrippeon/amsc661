#%%
import distmesh
import numpy as np
import matplotlib.pyplot as plt

# note that I modified the definitions of these functions in distmesh.py so that they behave 
# the way you would expect (as functions that return functions), rather than their previous behavior.

#%% build the SDF for the "L" shape

dL= distmesh.dunion(
    distmesh.drectangle(0, 1, 0, 0.5),
    distmesh.drectangle(0, 0.5, 0, 1)
)
pfix_L= np.array([
    [0,0],
    [0,1],
    [0.5,1],
    [0.5,0.5],
    [1,0.5],
    [1,0]
])

bbox_L= (0,1,0,1)
#%% build the SDF for the circle with holes

dHoles= distmesh.ddiff(
    distmesh.dcircle(0, 0, 2),
    distmesh.dunion(
        distmesh.dhalfplane((0,0), (0,1)),
        distmesh.dcircle(-1, -3/4, 1/2),
        distmesh.dcircle(1, -3/4, 1/2)
    )
)
pfix_Holes= np.array([
    [-2,0],
    [2,0]
])
bbox_Holes= (-2,2,-2,0)
#%% build the SDF for the pentagon
s= 5
theta= np.linspace(0, 2 * np.pi, s+1)
theta= theta[:-1] #we doubled the last point

#outer
xs= np.cos(theta)
ys= np.sin(theta)
ps_outer= np.stack((xs,ys),axis=1)
dP1= distmesh.dconvex(ps_outer)

#inner
phi= 1/2 * (1 + np.sqrt(5))
xs= (2 - phi) * np.cos(theta + np.pi)
ys= (2 - phi) * np.sin(theta + np.pi)
ps_inner= np.stack((xs,ys),axis=1)
dP2= distmesh.dconvex(ps_inner)

dPent= distmesh.ddiff(dP1, dP2)

pfix_Pent= np.concatenate([ps_inner, ps_outer], axis=0)

bbox_Pent= (-1,1,-1,1)
#%% mesh the areas 
h0=0.1
def fh(p):
    p= np.array(p)
    n, _= p.shape
    return np.ones(n)

pts_L, tri_L= distmesh.distmesh2D(
    fd=dL,
    fh=fh,
    h0=0.1,
    bbox=bbox_L,
    pfix=pfix_L
)
fig1, ax1= distmesh.visualize_result(pts_L, tri_L)
plt.show()
pts_Holes, tri_Holes= distmesh.distmesh2D(
    fd=dHoles,
    fh=fh,
    h0=0.2, # makes up for the physically larger size of this shape
    bbox=bbox_Holes,
    pfix=pfix_Holes
)
fig2, ax2= distmesh.visualize_result(pts_Holes, tri_Holes)
plt.show()
pts_Pent, tri_Pent= distmesh.distmesh2D(
    fd=dPent,
    fh=fh,
    h0=0.15,
    bbox=bbox_Pent,
    pfix=pfix_Pent
)
fig3, ax3= distmesh.visualize_result(pts_Pent, tri_Pent)
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt

# dimensions of the domain 
Nx= 100
Ny= 100

xmin= -np.pi
xmax= np.pi
ymin= 0
ymax= 2

hx= (xmax - xmin) / Nx
hy= (ymax - ymin) / Ny

x= np.linspace(xmin, xmax, Nx)
y= np.linspace(ymin, ymax, Ny)
X,Y= np.meshgrid(x,y)

# We will solve AU = F for U, which will be our solution vector

#%%## Construct A

# define the matrix. each block corresponds to a column of our domain
alpha= 2 * (1 / hx**2 + 1 / hy**2)
beta= 1 / hx**2
gamma= 1 / hy**2

# T goes on the diagonal blocks, comes from terms in the same column
T= np.diag(np.full(Nx, -alpha)) + np.diag(np.full(Nx-1, beta), 1) + np.diag(np.full(Nx-1, beta), -1)
# these are needed because of the periodic boundary condition
T[-1,0]= beta
T[0,-1]= beta
Ix= np.eye(Nx)
Iy= np.eye(Ny)

# this controls the layout of the terms from neighboring columns
M= np.diag(np.full(Ny-1, 1), 1) + np.diag(np.full(Ny-1, 1), -1)
# The two in this position comes from the adiabatic boundary condition
# we put a ghost point off the end of the domain, and assert it is equal to that one step into the domain
# this makes the partial derivative with respect to y here 0
M[-1,-2]= 2

# this is the final matrix
A = np.kron(Iy, T) + np.kron(M, gamma * Ix)

#%%## Construct f

f= -np.cos(X)
f[np.abs(X) > np.pi / 2]= 0
F= f.flatten()


#%% Solve

u= np.linalg.solve(A,F)
U= u.reshape(X.shape)

# %%
fig, ax= plt.subplots()
img= ax.imshow(U, cmap='magma', origin='lower')
ax.set(
    xlabel='x',
    ylabel='y',
    title='Solution to $\Delta u = -f$'
)
fig.colorbar(img)

plt.show()

# plot the result on the surface of a cylinder
R= 1

vX= R * np.cos(X)
vY= R * np.sin(X)
vZ= Y

norm= plt.Normalize(U.min(), U.max())

colors= plt.get_cmap('magma')(norm(U))
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(vX, vY, vZ, facecolors=colors)
ax.set(
    xlabel='x',
    ylabel='y',
    zlabel='z',
    title='Solution to $\Delta u = -f$'
)
plt.show()
# %%

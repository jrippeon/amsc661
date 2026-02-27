# %%
import numpy as np
import matplotlib.pyplot as plt
from time import process_time
from stiff import stiff

#%%########################### Robertson's problem #############################

a= 0.04
b= 1e4
c= 3e7

def f(t,u):
    x,y,z= u
    uprime= np.array([
        -a*x + b*y*z,
        a*x - b*y*z - c*y**2, 
        c*y**2 
    ])
    return uprime

def dfdy(t,u):
    x,y,z= u
    J= np.array([
        [-a, b*z, b*y],
        [a, -b*z - 2*c*y, -b*y],
        [0, 2*c*y, 0]
    ])
    return J

Tmax=100
y0= np.array([1.0, 0.0, 0.0])

#%%#############################################################################

methods= ['DIRK2', 'DIRKo3', 'BDF2']
h_values= [1e-1, 1e-2, 1e-3]

dim_names=['$x$', '$y$', '$z$']


times= np.zeros((len(h_values), len(methods)))

ts= {}
us= {}

for m_idx,method in enumerate(methods):
    for h_idx, h in enumerate(h_values):
        print(f'Method:{method}\th:{h}')
        t_start= process_time()
        t, u= stiff(
            func=f,
            jacobian=dfdy,
            t_span=[0,Tmax],
            y0=y0,
            h=h,
            method=method,
        )
        t_end= process_time()

        times[m_idx,h_idx]= t_end - t_start

        ts[(method, h)]= t
        us[(method, h)]= u
#%%
markers= ['+', 'x', '.']
fig, ax= plt.subplots(len(h_values), len(y0))
for m_idx,method in enumerate(methods):
    for h_idx, h in enumerate(h_values):
        t= ts[(method, h)]
        u= us[(method, h)]
        for dim_idx in range(len(y0)):
            # # only plot one point a second
            # density= int(1/h)
            # ax[dim_idx, h_idx].plot(t[::density], u[::density,dim_idx], marker=markers[m_idx])
            ax[dim_idx, h_idx].plot(t, u[:,dim_idx])

            # label time at the bottom
            if dim_idx == len(y0)-1:
                ax[dim_idx, h_idx].set(
                    xlabel='$t$'
                )
            # variables on the side
            if h_idx == 0:
                ax[dim_idx, h_idx].set(
                    ylabel=dim_names[dim_idx]
                )
            # h on the top
            if dim_idx == 0:
                ax[dim_idx, h_idx].set(
                    title=f'$h={h}$'
                )
# handles, labels= ax[0,0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
ax[0, 2].legend(methods)
plt.show()
# %%

fig, ax= plt.subplots()
for i, method in enumerate(methods):
    ax.loglog(h_values, times[i, :], label=method)
ax.legend(methods)
ax.set(
    xlabel='$h$',
    ylabel='Runtime [s]'
)

# %%

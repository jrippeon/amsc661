#%%
import numpy as np
import matplotlib.pyplot as plt
from rk import dirk
from time import process_time

#%% van der Pol oscillator
mu= 1e6
Tmax= 2e6
h= 1e-3
y0= np.array([2, 0])

def f(t,u):
    x,y= u
    return np.array([
        y,
        mu * (1 - x**2)*y - x
    ])
def dfdy(t,u):
    x,y= u
    return np.array([
        [0, 1],
        [-2 * mu * x * y - 1, mu * (1 - x**2)]
    ])


#%% calculate 
h=1e-4
adaptive=True

t_start= process_time()
t,u,h= dirk(
    f=f,
    dfdy=dfdy,
    t_span=[0,Tmax],
    y0=y0,
    method='DIRK2',
    h=h,
    adaptive=adaptive,
    progress=1e5,
    output_hist=True
)
t_end= process_time()
print(f'Executed in {t_end-t_start}s')
# %%
# plot x and y vs t
fig1, ax1= plt.subplots(1, 2)
ax1[0].plot(t, u[:,0])
ax1[0].set(
    xlabel='t',
    ylabel='x'
)
ax1[1].plot(t, u[:,1])
ax1[1].set(
    xlabel='t',
    ylabel='y'
)
# %%
# plot x vs y
fig2, ax2= plt.subplots()
ax2.plot(u[:,0], u[:,1])
ax2.set(
    xlabel='$x$',
    ylabel='$y$'
)

# %%
# plot step length
fig3, ax3= plt.subplots()
# ax3.scatter(u[:,0], u[:,1],c=np.log(h))

# ax3.scatter(t, u[:,0],c=np.log(h))

ax3.scatter(t, u[:,1],c=np.log(h))

# %%

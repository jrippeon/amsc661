# %%
from rk import dirk
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# use this to make filenames unique
current_time= datetime.now().strftime('%Y%m%d%H%M%S')


# %% Prothero-Robinson problem

L= 1e4
Tmax= 10
phi= lambda t: np.sin(t + np.pi/4)
phidot= lambda t: np.cos(t + np.pi/4)

f= lambda t,y: -L *(y - phi(t)) + phidot(t)
dfdy= lambda t,y: -L
y0= np.sin(np.pi/4) + 10


# %% try the method for a bunch of step sizes, then report the error (in the infinity norm)
# this takes a horrifically long time to run
p_values= np.arange(1, 6, 5/24)
e_max= np.zeros_like(p_values)

method='DIRK2'
for i, p in enumerate(p_values):
    print(f'Trying p={p}')
    h= 10**(-p)
    t, u= dirk(
        f=f, 
        t_span=[0, Tmax], 
        y0=y0, 
        method=method,
        h=h,
        dfdy=dfdy
    )
    y= np.exp(-L * t) * (y0 - phi(0)) + phi(t)
    e= np.abs(u - y)
    e_max[i]= np.max(e)
    np.savetxt(f'{method}_{current_time}.csv', e_max, delimiter=',')

# %% Plot some of the results for different step sizes
p_values= [1,2,3]
methods= ['DIRK2', 'DIRKo3']
Tmax=1
fig, ax= plt.subplots(2, len(p_values))
fig2, ax2= plt.subplots(2, len(p_values))
fig.suptitle('Errors $|e(t)|$')
fig2.suptitle('Solutions $u(t)$')
for i, method in enumerate(methods):
    for j, p in enumerate(p_values): 
        h= 10**(-p)
        t, u= dirk(
            f=f, 
            t_span=[0, Tmax], 
            y0=y0, 
            method=method,
            h=h,
            dfdy=dfdy
        )
        y= np.exp(-L * t) * (y0 - phi(0)) + phi(t)
        e= np.abs(u - y)

        ax[i,j].semilogy(t, np.abs(e))
        ax[i,j].set(
            title=f'{method}, $h=10^\u007b-{p}\u007d$'
        )
        if i == 1:
            ax[i,j].set(
                xlabel='t'
            )
        if j == 0:
            ax[i,j].set(
                ylabel='$|e(t)|$'
            )

        ax2[i,j].plot(t,u)
        ax2[i,j].set(
            title=f'{method}, $h=10^\u007b-{p}\u007d$'
        )
        if i == 1:
            ax2[i,j].set(
                xlabel='t'
            )
        if j == 0:
            ax2[i,j].set(
                ylabel='$u(t)$'
            )
# %%

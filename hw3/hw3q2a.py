# %%
from rk import dirk
import numpy as np
import matplotlib.pyplot as plt


# %% Prothero-Robinson problem

L= 1e4
Tmax= 10
phi= lambda t: np.sin(t + np.pi/4)
phidot= lambda t: np.cos(t + np.pi/4)

f= lambda t,y: -L *(y - phi(t)) + phidot(t)
dfdy= lambda t,y: np.array([[-L]])
y0= np.sin(np.pi/4) + 10


# DIRK2
p_values= np.arange(1, 6, 5/24)
# p_values= [5]
e_DIRK= np.zeros_like(p_values)

# try the method for a bunch of step sizes, then report the error (in the infinity norm)
for i, p in enumerate(p_values):
    # print(f'Trying p={p}')
    h= 10**(-p)
    t, u= dirk(
        f=f, 
        t_span=[0, Tmax], 
        y0=y0, 
        method='DIRKo3',
        h=h,
        dfdy=dfdy
    )
    y= np.exp(-L * t) * (y0 - phi(0)) + phi(t)
    e= np.abs(u - y)
    e_max= np.max(e)
    e_DIRK[i]= e_max
    np.savetxt('DIRKo3_bad.csv', e_DIRK, delimiter=',')

# %%
import numpy as np
import matplotlib.pyplot as plt

# %% First Plot (good initial condition)
e2= np.loadtxt('DIRK2_20260223212928.csv')
e3= np.loadtxt('DIRKo3_v3.csv')
p= np.arange(1, 6, 5/24)
h= 10**(-p)

fig, ax= plt.subplots(1, 2)
ax[0].loglog(h,e2)
ax[0].set(
    xlabel='$h$',
    ylabel='Error',
    title='DIRK2'
)
ax[0].loglog(h, 3.4e-5*(h), color='r', linestyle='--')
ax[0].loglog(h, 3.0e-2*(h**2), color='k', linestyle='--')
ax[0].legend(['Observed', 'O(h)','$O(h^2)$'])


ax[1].loglog(h,e3)
ax[1].set(
    xlabel='$h$',
    ylabel='Error',
    title='DIRKo3'
)
ax[1].loglog(h, 1.0e-1*(h**2), color='r', linestyle='--')
ax[1].loglog(h, 9e2*(h**3), color='k', linestyle='--')
ax[1].legend(['Observed', '$O(h^2)$','$O(h^3)$'])
plt.show()

# %% Second plot (bad initial condition)
e2b= np.loadtxt('DIRK2_bad.csv')
e3b= np.loadtxt('DIRKo3_bad.csv')

p= np.arange(1, 6, 5/24)
h= 10**(-p)

fig, ax= plt.subplots(1, 2)
ax[0].loglog(h,e2b)
ax[0].set(
    xlabel='$h$',
    ylabel='Error',
    title='DIRK2'
)
# ax[0].loglog(h, 3.4e-5*(h), color='r', linestyle='--')
ax[0].loglog(h, 1.5e7*(h**2), color='k', linestyle='--')
ax[0].legend(['Observed', '$O(h^2)$'])


ax[1].loglog(h,e3b)
ax[1].set(
    xlabel='$h$',
    ylabel='Error',
    title='DIRKo3'
)
# ax[1].loglog(h, 1.0e-1*(h**2), color='r', linestyle='--')
ax[1].loglog(h, 4e11*(h**3), color='k', linestyle='--')
ax[1].legend(['Observed', '$O(h^3)$'])
plt.show()
# %%

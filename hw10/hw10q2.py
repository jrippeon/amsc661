import numpy as np
import matplotlib.pyplot as plt
import hyperbolic as hb
from scipy.sparse import diags

# paratemers ###################################################################

xmin, xmax= -1, 10
tmax= 5.5
h= .03
k= 0.005

def u0(x):
    # * is logical and lol
    return 2 * (x<=1) * (x>=0) + 1 * (x>1) * (x<=2)

f= lambda u: 0.5 * u**2
# this is true since it's monotonically increasing for positive u
fmin= lambda uL,uR: min(f(uR), f(uL))
fmax= lambda uL, uR: max(f(uL), f(uR))

# compute results ##############################################################

results= {}

for i, update in enumerate([hb.lax_fr_update, hb.richtmeyer_update, hb.maccormack_update]):
    t, x, u= hb.numerically_solve(
        f=f,
        update=update,
        xmin=xmin,
        xmax=xmax,
        tmax=tmax,
        h=h,
        k=k,
        u0=u0
    )
    results[i]=(t,x,u)

t,x,u= hb.godunoff(
    f=f,
    fmin=fmin,
    fmax=fmax,
    xmin=xmin,
    xmax=xmax,
    tmax=tmax,
    h=h,
    k=k,
    u0=u0
)
results[3]= (t,x,u)

# display results ##############################################################

def plot_at_time(t,x,u,t_plot,ax):
    n_plot= np.argmin(np.abs(t[:, None] - t_plot))
    print(f'Timestep {n_plot} is closest to t={t_plot}, with t[n]={t[n_plot]}.')
    ax.plot(x, u[n_plot])

method_names= ['Lax Friedrichs', 'Richtmeyer', 'MacCormack', 'Godunoff']

index= 3

times= np.array([0, 0.5, 1.5, 2.5, 3.5, 5])
fig, ax= plt.subplots()

for t_plot in times:
    plot_at_time(*results[index], t_plot, ax)
ax.set(
    xlabel='x',
    ylabel='u(x,t)',
    title=f'{method_names[index]} Method'
)
ax.legend([f't={t}' for t in times])
plt.show()
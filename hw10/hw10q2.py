import numpy as np
import matplotlib.pyplot as plt
import hyperbolic as hb
from scipy.sparse import diags

# paratemers ###################################################################

xmin, xmax= -1, 10
tmax= 5.5
h= .025
k= 0.005

def u0(x):
    # * is logical and lol
    return 2 * (x<=1) * (x>=0) + 1 * (x>1) * (x<=2)

def u_exact(x,t):
    if 0 <= t and t < 1:
        return 0 * (x<0) + x/t * (0<=x)*(x<2*t) + 2 * (2*t<=x)*(x<3/2 * t + 1) + 1 * (x >= 3/2 * t + 1) * (x < 1/2 * t + 2)
    elif 1 <= t and t < 3/2:
        return 0 * (x<0) + x/t * (0<=x)*(x<2*t) + 2 * (2*t<=x)*(x < t + 3/2) + 0 *(x >= t + 3/2)
    else:
        return 0 * (x<0) + x/t * (0<=x)*(x<np.sqrt(6 * t)) + 0 * (x<np.sqrt(6 * t))



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

t,x,u= hb.godunov(
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

def plot_at_time(t,x,u,t_plot,ax, label=None):
    n_plot= np.argmin(np.abs(t[:, None] - t_plot))
    print(f'Timestep {n_plot} is closest to t={t_plot}, with t[n]={t[n_plot]}.')
    if label:
        ax.plot(x, u[n_plot], label=label)
    else:
        ax.plot(x, u[n_plot])

method_names= ['Lax Friedrichs', 'Richtmeyer', 'MacCormack', 'Godunoff']

for index in range(4):
    times= np.array([0, 0.5, 1.5, 2.5, 3.5, 5])
    fig, ax= plt.subplots()

    for t_plot in times:
        t, x, u= results[index]
        plot_at_time(t, x, u, t_plot, ax, label=f't={t_plot}')
        # the _ apparently makes it not appear in the legend, neat
        ax.plot(x, u_exact(x, t_plot), c='k', linestyle='--', alpha=0.5)
    ax.set(
        xlabel='x',
        ylabel='u(x,t)',
        title=f'{method_names[index]} Method'
    )
    ax.legend()
    # ax.legend([f't={t}' for t in times])
    plt.show()
# %%
from verlet import verlet
import numpy as np
import matplotlib.pyplot as plt

# %%
p0= np.array([0, 1/2])
q0= np.array([2, 0])

a= 4/3
period= 2 * np.pi * a**(3/2)

# I want this to be able to handle a (N,d) array of N different p values
def T(p):
    if p.ndim == 1:
        return 1/2 * np.linalg.norm(p)**2
    else:
        return 1/2 * np.linalg.norm(p, axis=1)**2
def U(q):
    if q.ndim == 1:
        return - 1 / np.linalg.norm(q)
    else:
        return - 1 / np.linalg.norm(q, axis=1)

dT= lambda p: p
def dU(q):
    if q.ndim == 1:
        return q / (np.linalg.norm(q))**3
    else:
        return q / (np.linalg.norm(q, axis=1))**3


t_span= [0, 10*period]
h= 0.01 * period

t, p, q= verlet(dT, dU, p0, q0, t_span, h)


# %%
fig1, ax1= plt.subplots()
ax1.plot(q[:,0], q[:,1])
ax1.set_aspect('equal')
ax1.set(
    xlabel='x',
    ylabel='y',
    title='Ten orbits integrated with Stoermer-Verlet'
)
plt.show()

# %%
H= T(p) + U(q)
fig2, ax2= plt.subplots()
ax2.plot(t, H)
ax2.set(
    title='Total energy vs. time',
    xlabel='$t$',
    ylabel='$H(p(t), q(t))$'
)
plt.show()
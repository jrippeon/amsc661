from verlet import verlet
import numpy as np
import matplotlib.pyplot as plt

p0= np.array([0, 1/2])
q0= np.array([2, 0])

a= 4/3
period= 2 * np.pi * a**(3/2)

# T=  lambda p: 1/2 * p.T @ p
# U=  lambda q: 1 / np.sqrt(q.T @ q)

# dT= lambda p: p
# dU= lambda q: q / (q.T @ q)**(3/2)

# using np.linalg.norm should make it more vectorization-friendly :D
T=  lambda p: 1/2 * np.linalg.norm(p)
U=  lambda q: 1 / np.sqrt(np.linalg.norm(q))

dT= lambda p: p
dU= lambda q: q / (np.linalg.norm(q))**3

t_span= [0, 10*period]
h= 0.01 * period

t, p, q= verlet(dT, dU, p0, q0, t_span, h)



plt.plot(q[:,0], q[:,1])
plt.show()


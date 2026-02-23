from rk import plot_RAS
import numpy as np
import matplotlib.pyplot as plt

gamma1= 1 + np.sqrt(2)/2
gamma0= 1 - np.sqrt(2)/2
A1= np.array([
    [gamma1, 0],
    [1 - 2 * gamma1, gamma1]
])
A0= np.array([
    [gamma0, 0],
    [1 - 2 * gamma0, gamma0]
])
b= np.array([1/2, 1/2])

fig, axs= plt.subplots(1,2)

plot_RAS(
    A=A0,
    b=b,
    window_size=15,
    filled=True,
    ax=axs[0],
    decorations=True
)
plot_RAS(
    A=A1,
    b=b,
    window_size=15,
    filled=True,
    ax=axs[1],
    decorations=True
)
axs[0].set(
    title=r'$\gamma = 1 - \frac{\sqrt{2}}{2}$'
)
axs[1].set(
    title=r'$\gamma = 1 + \frac{\sqrt{2}}{2}$'
)

plt.show()
import matplotlib.pyplot as plt
import numpy as np

############################ Solver Implementation #############################

def f(s):
    x,y,u,v= s
    return np.array([
        u,
        v,
        -x / (x**2 + y**2),
        -y / (x**2 + y**2)
    ])

# midpoint rule with Forward Euler predictor

s0= np.array([1, 0, 0, 1])
Ns= [20, 40, 80]

results= {}
for N in Ns:
    h= 2 * np.pi / N

    # "s" for "states", four periods in this case
    s= np.zeros((4 * N, 4))

    s[0]= s0

    for n in range(1, len(s)):
        # predict half a step ahead with Forward Euler
        s_star= s[n-1] + h/2 * f(s[n-1])
        # apply the midpoint rule
        s[n]= s[n-1] + h * f(s_star)
    results[N]= s

############################## Display Results #################################

fig, ax= plt.subplots()
for N in Ns:
    s= results[N]
    ax.plot(s[:, 0], s[:, 1])
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('Midpoint Rule Solution')
ax.legend(Ns, loc='upper right')
ax.set_aspect('equal')
plt.show()
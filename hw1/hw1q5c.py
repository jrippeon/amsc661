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

# ad-hoc linear two-step explicit method (4th order)

# computed constants from part (a) and givens
a0= -4
a1= 5
b0= 4
b1= 2
s0= np.array([1, 0, 0, 1])
Ns= [20, 40, 80]

results= {}
for N in Ns:
    h= 2 * np.pi / N

    # "s" for "states", two periods worth
    s= np.zeros((2 * N, 4))

    s[0]= s0

    # use Forward Euler to obtain s_1 
    # (we need two points as an initial condition for our two-step method)
    s[1]= s[0] + h * f(s[0])

    for n in range(2,len(s)):
        # update rule described in question 5
        s[n]= a0 * s[n-1] + a1 * s[n-2] + h * (b0 * f(s[n-1]) + b1 * f(s[n-2]))
    
    results[N]= s


############################## Display Results #################################

# print a table of N vs final norm
print('\\begin{tabular}{c | c}\n\\hline \n\\(N\\) & \\(\\| s(4 \\pi) \\|\\)\\\\')
for N in Ns:
    s= results[N]
    # we only care about the norm of the x and y components
    norm= np.linalg.norm(s[-1, 0:2])
    print(f'\t{N} & {norm:.4g} \\\\')
print('\\end{tabular}')

# plot of N vs final norm
fig, ax= plt.subplots()
ax.semilogy(Ns, [np.linalg.norm(results[N][-1, 0:2]) for N in Ns], marker='o')
ax.set_xticks(Ns)
ax.set_xlabel('$N$')
ax.set_ylabel('$\| s(4 \pi) \|$')
ax.set_title('Final Norm vs $N$')
plt.show()

# plot of trajectory (basically useless lol)
fig, ax= plt.subplots()
for N in Ns:
    s= results[N]
    ax.plot(s[:, 0], s[:, 1])
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('Linear 2-Step Explicit 4th-Order Method Solution')
ax.legend(list(reversed(Ns)))
ax.set_aspect('equal')
plt.show()
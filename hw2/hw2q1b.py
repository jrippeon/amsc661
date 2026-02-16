# %%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from time import process_time

# %% system
# mu is the mass of the moon, (1-mu) the mass of the earth
mu= 0.012277471

def f(t, y):
    y1, y2, y3, y4= y

    r1= ((y1 + mu)**2 + y2**2)**(1/2)
    r2= ((y1 - 1 + mu)**2 + y2**2)**(1/2)

    dy1= y3
    dy2= y4
    dy3= y1 + 2*y4 - (1-mu)*(y1 + mu)/ r1**3 - mu * (y1 - 1 + mu) / r2**3
    dy4= y2 - 2*y3 - (1-mu)*y2 / r1**3 - mu * y2 / r2**3

    return np.array([dy1, dy2, dy3, dy4])

# absurd precision lol
y0= np.array([0.994, 0, 0, -2.00158510637908252240537862224])
# period
T= 17.0652165601579625588917206249
epsilon= 10**-12

# %%  one period with DOPRI5(4) (AKA RK45)
one_period_result= solve_ivp(
    fun= f,
    t_span=[0, T],
    y0= y0,
    method='RK45',
    rtol= epsilon,
    atol=epsilon
)    


# %% multiple periods with different methods
Tmax= 100

# RK45
print('Starting RK45')
start= process_time()
RK45_solution= solve_ivp(
    fun= f,
    t_span=[0, Tmax],
    y0= y0,
    method='RK45',
    rtol= epsilon,
    atol=epsilon
)    
stop= process_time()
RK45_time= stop - start

# DOP853
print('Starting DOP853')
start= process_time()
DOP853_solution= solve_ivp(
    fun= f,
    t_span=[0, Tmax],
    y0= y0,
    method='DOP853',
    rtol= epsilon,
    atol=epsilon
)    
stop= process_time()
DOP853_time= stop - start

# Radau
print('Starting Radau')
start= process_time()
Radau_solution= solve_ivp(
    fun= f,
    t_span=[0, Tmax],
    y0= y0,
    method='Radau',
    rtol= epsilon,
    atol=epsilon
)    
stop= process_time()
Radau_time= stop - start

# %%  plot the result over one period
fig1, ax1= plt.subplots()
ax1.plot(one_period_result.y[0], one_period_result.y[1])
ax1.set(
    xlabel='$y_1$',
    ylabel='$y_2$',
    title='One Period of Arenstorf Orbit'
)
plt.show()

# %% plot the orbits for each solver
fig2, ax2= plt.subplots()
ax2.plot(RK45_solution.y[0], RK45_solution.y[1])
ax2.plot(DOP853_solution.y[0], DOP853_solution.y[1])
ax2.plot(Radau_solution.y[0], Radau_solution.y[1])
ax2.legend(['RK45', 'DOP853', 'Radau'])
ax2.set(
    title='Arenstorf Orbit Solutions for Several Methods',
    xlabel='$y_1$',
    ylabel='$y_2$'
)
plt.show()

fig3, ax3= plt.subplots()
ax3.bar(
    x=['RK45', 'DOP853', 'Radau'],
    height=[RK45_time, DOP853_time, Radau_time]
)
ax3.set(
    ylabel='Time [s]',
    title='Time Comparison of Various Methods'
)
plt.show()
# %% plot the orbits for each solver on separate plots
fig4, ax4= plt.subplots(2,2)
fig4.delaxes(ax4[1,1])
ax4[0, 0].plot(RK45_solution.y[0], RK45_solution.y[1])
ax4[0, 0].set(
    xlabel='$y_1$',
    ylabel='$y_2$',
    title='RK45'
)
ax4[0, 0].set_aspect('equal')

ax4[0,1].plot(DOP853_solution.y[0], DOP853_solution.y[1])
ax4[0,1].set(
    xlabel='$y_1$',
    ylabel='$y_2$',
    title='DOP853'
)
ax4[0,1].set_aspect('equal')

ax4[1,0].plot(Radau_solution.y[0], Radau_solution.y[1])
ax4[1,0].set(
    xlabel='$y_1$',
    ylabel='$y_2$',
    title='Radau'
)
ax4[1,0].set_aspect('equal')

# %%

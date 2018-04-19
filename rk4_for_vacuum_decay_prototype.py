import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

F = 0
L = 0
epsilon = 0

def constant(x):
    return 1

def integrate(function, lower_limit, upper_limit, iterations):
    step = (upper_limit - lower_limit) / iterations
    result = 0
    x = lower_limit
    for i in range(iterations):
        result += step * function(x)
        x += step
    return result

def thin_wall_higgs_potential(x):
    return (L / 24) * pow(pow(x, 2) - pow(F, 2), 2) - (epsilon / (2 * F)) * (x + F)

def thin_wall_higgs_potential_derivative(x):
    return (L / 6) * x * (pow(x, 2) - pow(F, 2)) + (epsilon / (2 * F))

def higgs_potential(x):
    return (L / 24) * pow(pow(x, 2) - pow(F, 2), 2)

def potential(x):
    return np.sqrt(2 * higgs_potential(x))

def dx(t, x):
    return potential(x)

def rk4(t, x, derivative, step): # For solving a differential equation of the form dx/dt with a step in t of value 'step'
    k_1 = derivative(t, x)
    k_2 = derivative(t + (step/2), x + (step/2) * k_1)
    k_3 = derivative(t + (step/2), x + (step/2) * k_2)
    k_4 = derivative(t + step, x + step * k_3)
    return x + (step/6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)

def instanton_solution(t):
    return F * np.tanh(np.sqrt(L / 24) * (t - T_1))

T_NEGATIVE_LIMIT = -4
T_POSITIVE_LIMIT = 4
T_INTERVAL = T_POSITIVE_LIMIT - T_NEGATIVE_LIMIT
T_POINTS = 100000
T_STEP = T_INTERVAL / T_POINTS
T_1 = 0
F = 1
L = 24
epsilon = 2
deviation = 0.001
t_0, x_0 = T_NEGATIVE_LIMIT, -1 * F + deviation # Starting point, starts at the right hand minima located at phi = 1 and therefor x should be at the positive boundary
t, x = [], []
t = np.zeros(T_POINTS)
t[0] = t_0
X = np.zeros(T_POINTS)
X[0] = x_0
inst = np.zeros(T_POINTS)
inst[0] = instanton_solution(t_0)
print("values = %s, %s" %(t[0], X[0]))

x_const = np.zeros(T_POINTS)
potential_plot = np.zeros(T_POINTS)

x_const[0] = T_NEGATIVE_LIMIT
potential_plot[0] = 1 * thin_wall_higgs_potential(x_const[0])

for i in tqdm(range(1, T_POINTS)):
    t[i] = t[i-1] + T_STEP
    X[i] = rk4(t[i-1], X[i-1], dx, T_STEP)
    inst[i] = instanton_solution(t[i])
    x_const[i] = x_const[i-1] + T_STEP
    potential_plot[i] = 1 * thin_wall_higgs_potential(x_const[i])

fig2 , ax2=plt.subplots(2,1)
ax2[0].set_xlim([T_NEGATIVE_LIMIT, T_POSITIVE_LIMIT])
#ax2[0].set_ylim([0, 1])
ax2[0].set_title('x configuration')
ax2[0].set_xlabel('t')
ax2[0].set_ylabel('x')
ax2[1].set_ylim([-1 * epsilon - 0.2, 0.4])
#ax2[1].set_ylim([-0.1, epsilon + 0.2])
ax2[1].set_title('Potential')
ax2[1].set_xlabel('x')
ax2[1].set_ylabel('V')
ax2[0].plot(t, X, label='rk4 Solution')
ax2[0].plot(t, inst, label='Analytic solution')
ax2[1].plot(x_const, potential_plot, label='Potential')
ax2[0].legend()
ax2[1].legend()
plt.show()

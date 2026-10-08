from scipy.optimize import linprog

import sympy
from sympy import Matrix
from sympy import ImmutableMatrix

# Standard form introduces 2 slack variables
# (2  5 -1 0)                     (24)
# (1 -2 0 -1) * [x, s1, s2] = (24)
# x >= 0, s_1 >= 0, s_2 >= 0, s_3 >= 0 -- Ignore the last 3

A_eq = ImmutableMatrix([[2,  5, -1, 0],
                        [1, -2, 0, -1]])
b_eq = ImmutableMatrix([60, 0])
c_prime = Matrix([60, 200, 0, 0]) # dc 'bout the last 3
standard_bounds = [(0, None), (0, None), (0, None), (0, None)]

standard_solution = linprog(c=c_prime, A_eq=A_eq, b_eq=b_eq, bounds=standard_bounds)
print(standard_solution) # Yields an optimal feasible degenerate solution (30, 0, 0, 0)

# Polyhedron for any standard form equation exists in n-m-dimensional space the
# other dimensions come out degenerate. n-m = 2 for this problem

import numpy as np
import matplotlib.pyplot as plt

# Create a grid for plotting
x = np.linspace(0, 200, 200)
y1 = 12 - 0.4*x
y2 = 0.5*x

# Plotting
plt.figure()
plt.plot(x, y1, label='2x + 5y >= 60')
plt.plot(x, y2, label='x - 2y >= 0')
plt.xlim((0, 40))
plt.ylim((0, 40))
plt.fill_between(x, np.minimum(y1, y2), color='gray', alpha=0.5)

# Mark the optimal solution
plt.plot(standard_solution.x[0], standard_solution.x[1], 'ro')  # Optimal point
plt.title('Polyhedron & Solution')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()

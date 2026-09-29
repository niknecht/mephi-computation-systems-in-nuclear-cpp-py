from scipy.optimize import linprog

import sympy
from sympy import Matrix
from sympy import ImmutableMatrix

# General form problem (see "Introduction to Linear Optimization", MIT)
# (2 1)        (24) (weird vector relation notation ik, the whole book is weird tbh)
# (1 2) * x <= (24)  or A*x <= b
# (1 1)        (16)
# x >= 0
# maximize c'x where c' is the revenue vector

# Transmutations of the problem to get it to the standard form can be ommited thanks
# to scipy capabilities. Only required one is the cost must be minimized
A_ub = ImmutableMatrix([[2, 1],
                        [1, 2],
                        [1, 1]])
b = ImmutableMatrix([24, 24, 16])
c = ImmutableMatrix([-2, -3])
bounds = [(0, None), (0, None)]

solution = linprog(c=c, A_ub=A_ub, b_ub=b, bounds=bounds)
print(solution) # Yields optimal feasible x = (8, 8)

# Geometric interpretation + solution requires standard form
# Standard form introduces 3 slack variables
# (2 1 1 0 0)                     (24)
# (1 2 0 1 0) * [x, s1, s2, s3] = (24)
# (1 1 0 0 1)                     (16)
# x >= 0, s_1 >= 0, s_2 >= 0, s_3 >= 0 -- Ignore the last 3

A_eq = ImmutableMatrix([[2, 1, 1, 0, 0],
                        [1, 2, 0, 1, 0],
                        [1, 1, 0, 0, 1]])
b_eq = ImmutableMatrix([24, 24, 16])
c_prime = Matrix([-2, -3, 0, 0, 0]) # dc 'bout the last 3
standard_bounds = [(0, None), (0, None), (0, None), (0, None), (0, None)]

standard_solution = linprog(c=c_prime, A_eq=A_eq, b_eq=b_eq, bounds=standard_bounds)
print(standard_solution) # Yields an optimal feasible degenerate solution (8, 8, 0, 0, 0)

# Polyhedron for any standard form equation exists in n-m-dimensional space the
# other dimensions come out degenerate. n-m = 2 for this problem

import numpy as np
import matplotlib.pyplot as plt

# Create a grid for plotting
x = np.linspace(0, 200, 200)
y1 = 24 - 2*x  # From x + y <= 5
y2 = 12 - 0.5*x  # From 2x + y <= 8
y3 = 16 - x

# Plotting
plt.figure()
plt.plot(x, y3, label='x + y <= 16')
plt.plot(x, y1, label='2x + y <= 24')
plt.plot(x, y2, label='x + 2y <= 24')
plt.xlim((0, 20))
plt.ylim((0, 20))
plt.fill_between(x, np.minimum(y1, y2), color='gray', alpha=0.5)

# Mark the optimal solution
plt.plot(standard_solution.x[0], standard_solution.x[1], 'ro')  # Optimal point
plt.title('Polyhedron & Solution')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()

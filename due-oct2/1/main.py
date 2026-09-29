from scipy.optimize import linprog
import sympy
from sympy import Matrix
from sympy.vector import Vector
from sympy import ImmutableMatrix

# Notation as per "Inproducation to Linear Optimization", Bertsimas, Tsitsiklis, MIT
A = ImmutableMatrix([[2, 1],
                     [1, 3]])
#x = Matrix([n_a, n_b])
b = ImmutableMatrix([20, 30])
c_prime = ImmutableMatrix([3, 2])

# The problem (standard form in MIT notation)
# minimize c_prime*x
# subject to:
# A*x = b
# x > 0

# Non-standard noataion in scipy
bounds = [(0, None), (0, None)]

solution = linprog(c=c_prime, A_eq=A, b_eq=b, bounds=bounds)

print(solution) # This yields x = [6, 8]

# A has 2 linearly independent columns as a m=2 n=2 matrix,
# therefore B = A and the polyhedron exists in R^2

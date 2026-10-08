# This is just a copy-paste of the previous assignment

# I'm still using the same notation from the same book
# (that explains duality just terribly, I had to derive most of the stuff from scratch)

# Standard form problem:
# minimize -30w - 35b - 45s
# subject to
# w + b + s = 100
# w - s1 = 40 -- slack
# b + s2 = 50 -- surplus
# s + s3 = 20 -- surplus
# s, b, w, s1, s2, s3 > 0

# Or, which is equivalent:
# minimize
#           c           *           x
# (-30, -35, -45, 0, 0, 0) * (w, b, s, s1, s2, s3)
# subject to
#          A         *   x    =    b
# ( 1  1  1  0 0 0 )   ( w  )   ( 100 )
# ( 1  0  0 -1 0 0 )   ( b  )   (  40 )
# ( 0  1  0  0 1 0 ) * ( s  ) = (  50 )
# ( 0  0  1  0 0 1 )   ( s1 )   (  20 )
#                      ( s2 )
#                      ( s3 )

from scipy.optimize import linprog
from sympy import ImmutableMatrix
from sympy import Matrix

A_eq = ImmutableMatrix([[1, 1, 1,  0, 0, 0],
                        [1, 0, 0, -1, 0, 0],
                        [0, 1, 0,  0, 1, 0],
                        [0, 0, 1,  0, 0, 1]])
b_eq = ImmutableMatrix([100, 40, 50, 20])
c_prime = Matrix([-30, -35, -45, 0, 0, 0]) # dc 'bout the last 3
standard_bounds = [(0, None), (0, None), (0, None), (0, None), (0, None), (0, None)]

standard_solution = linprog(c=c_prime, A_eq=A_eq, b_eq=b_eq, bounds=standard_bounds)
print(standard_solution) # Yields an optimal feasible solution (40, 40, 20, 0, 10, 0) at the price of -3500

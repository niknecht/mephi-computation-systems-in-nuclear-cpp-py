from sympy import Matrix
from sympy import ImmutableMatrix

from scipy.optimize import linprog

# minimize
# ( 200 )   ( a )
# ( 150 ) * ( b )
# ( c_o )   ( o )
# subject to
# ( 0.3 0.20 0 )   ( a )    ( 30 )
# ( 0.1 0.15 1 ) * ( b ) >= ( 15 )
#                  ( o )
# a, b, o >= 0
#
# We could either find the optimal cost without fish oil
# and introduce an additional constraint that the new price, with the oil, must be equal (edge case of no more)
# Or we could look at a dual first
# The dual finds an optimal vec p from R^2 and an optimal price, which is going to be same as for the primal
# have 3 constraints for 2 variables, >= turns into <=, minimize - into maximize
#
# maximize
# (p1 p2) * (30 15)
# subject to
#            ( 0.3 0.20 0 )
# (p1 p2) *  ( 0.1 0.15 1 ) <= ( 200 150 c_o )
# p1, p2 >= 0
#
# This can be further rewritten as:
# maximize
# (p1 p2) * (30 15)
# subject to
#            ( 0.3 0.20 )
# (p1 p2) *  ( 0.1 0.15 ) <= ( 200 150 )
# p1, p2 >= 0
# p2 <= c_o
#
# it is now easy to see that p2 is the penalty to increasing oil requirement by one.
# If p2 is left with no upper bound, what we'll see is the price of increasing the total oil requirement by one
# fiven optimal choice of x in R^2. Obviously, to make the choice of fish oil optimal, it needs to match optimal
# choice for the other two foods.
#
# None of this was in the book though, so I did use a little AI.
#
# Now we're looking for c_o such that the penalty is maximum
# The only use of dual form here is that we've moved the parameter from cost to constraints

A_eq = ImmutableMatrix([[0.3, 0.20],
                        [0.1, 0.15]])
b_eq = ImmutableMatrix([200, 150])
c_prime = Matrix([-30, -15])
standard_bounds = [(0, None), (0, None)]

standard_solution = linprog(c=c_prime, A_eq=A_eq, b_eq=b_eq, bounds=standard_bounds)

print("Maximum price of fish oil is:" + str(int(standard_solution.x[1])))

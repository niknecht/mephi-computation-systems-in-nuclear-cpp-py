from sympy import Matrix
from sympy import ImmutableMatrix

from scipy.optimize import linprog

# minimize
# l -- optimal maximum of 2*a1 + 5*a2 and 3*b1 + 3*b2
# subject to
#                     ( a1 )
# ( 1 0 1 0 0 0 0 )   ( a2 )   (40)
# ( 0 1 0 1 0 0 0 ) * ( b1 ) = (20)
# ( 2 5 0 0 1 1 0 )   ( b2 )   ( 0)
# ( 0 0 3 3 1 0 1 )   (  l )   ( 0)
#                     ( s1 )
#                     ( s2 )
# a1, a2, b1. b2. l, s1, s2 >= 0
# Duality is not particularly useful here as we're not looking for the penalty
# on increasing any requirement by one unit

A = ImmutableMatrix([[1, 0, 1, 0,  0, 0, 0],
                     [0, 1, 0, 1,  0, 0, 0],
                     [2, 5, 0, 0, -1, 1, 0],
                     [0, 0, 3, 3, -1, 0, 1]])
c = ImmutableMatrix([0, 0, 0, 0, 1, 0, 0])
b = ImmutableMatrix([40, 20, 0 , 0])
ranges = [(0, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None)]

standard_solution = linprog(A_eq = A, b_eq = b, c = c, bounds = ranges)

# It's obvious that the optimal distribution of pages within a translator
# is inverse to the page/hour cooficient of those pages

print("Your distribution:\
      \nfirst:\
      \n  technical: " + str(int(standard_solution.x[0])) +
     "\n  legal: " + str(int(standard_solution.x[1])) +
     "\nsecond:\
      \n  technical: " + str(int(standard_solution.x[2])) +
     "\n  legal: " + str(int(standard_solution.x[3])) +
     "\nYour optimum: " + str(standard_solution.fun))

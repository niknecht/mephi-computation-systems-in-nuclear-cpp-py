# Okay so the last book covers none of this, so I'm switching to
# Jorge Nocedal, Stephen J. Wright's Numerical Optimization (2nd edition)
# (damn' good book btw)

# I am assuming the gradient descent method here is the steepest descent method
# A quite slow globally convergent according to Zoutendijk due to it's theta choice
# (i.e. choice of direction p_k and step alpha_k) numerical optimization algorithm.
# Zoutendijk here follows from the satisfaction of Wolfe/Goldstein at each step
# of the steepest descent

# (Zoutendijk, Godlstein and Wolfe are, ofc, forms of conditions on direction,
# i.e. on p_k, alpha_k and x_k or theta_k = arccos(fi(alpha_k)) = arccos(grad(f)_k*p_k/(||grad(f)_k||||p_k|))
# and fi(alpha_k) = f(x_k + alpha_k*p_k))

# (see theorem 3.2 of the book that proves
# For any "good" f: (i) f is lower bounded on R^n, (ii) f is C^1 in some N in R^n and
# (iii) grad(f) is Lipschitz continious:
# (i) && (ii) && (iii) && Wolfe/Goldstein => Zoutendijk)
# The proof for Wolfe is quite easy too

# Theorem 3.2 is an alternative approach to the Lagrange's method used in the lecture
# And yeah, the steepest descent is an evolution of what you call gradient descent
# That chooses etta instead of experimenting with its value.

# This is essential because it is the reasoning behind my algorithm implementation

# Steepest descent algorighm
# Variant 4: f = x^2 + xy + y^2 + e^x - 3y

# 1) f(x, y) is very obviously lower bounded (lim_(x,y->+-inf)(f) = +inf)
# 2) f(x, y) is ... hence it is C^1 on R^2
# 3) grad(f)(x, y) is x and y differentiable, which is the stronger condition to Lipschitz continious
# Wolfe conditions obviously are met
# cos(theta_k) is chosen to be -1 at each step, hence - global convergence

import sympy
import numpy

from scipy.optimize import brentq

x,y,alpha = sympy.symbols('x,y,alpha', real=True)

f_expr = x**2 + x*y + y**2 - 3*y + (2.73)**x
grad_f_expr = [ sympy.diff(f_expr, x), sympy.diff(f_expr, y) ]

f_eval = sympy.lambdify((x,y), f_expr, modules='numpy')
grad_f_eval = sympy.lambdify((x,y), grad_f_expr, modules='numpy')

df_dx_expr,df_dy_expr = grad_f_expr
xi = x - alpha*df_dx_expr
etta = y - alpha*df_dy_expr

f_next_prime_expr = sympy.diff(f_expr.subs({x:xi, y:etta}, simultaneous=True), alpha)
f_next_prime_eval = sympy.lambdify((x,y,alpha), f_next_prime_expr, modules='numpy')

def steepest_descent(p0, grad_f_tolerance=1e-8, maxIter=1000):
    points = [p0]

    for i in range(maxIter): #TODO: break cond; find p
        p = numpy.asarray(points[i-1], dtype=float)

        # Now find alpha that makes f next prime zero under these xi and etta of x, y and alpha
        # Prepping parameters for the func that searches for the optimal parameter value
        lo = 0.0 # The derivative by alpha at this alpha here is guarranteed by method negative
        hi = 1.0 # Max step len
        f_next_prime_of_alpha_eval = lambda alpha: float(f_next_prime_eval(*p, alpha))
        while(f_next_prime_of_alpha_eval(hi) < 0):
            hi*=2.0
        steepest_descent_alpha_eval_done = brentq(f_next_prime_of_alpha_eval, lo, hi)

        p = p - steepest_descent_alpha_eval_done*numpy.asarray(grad_f_eval(*p));
        points.append(p)

        if(numpy.linalg.norm(numpy.asarray(grad_f_eval(*p))) < grad_f_tolerance):
            break

    return points

# ------------------------- Actual programm -----------------------------

points = steepest_descent([0,0])

# [-1.2; 2.1] - minimum according to the method/theory above
print(points[-1], "\nMinimum")

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

x,y,alpha = sympy.symbols('x,y,alpha', real=True)

f_expr = x^2 + x*y + y^2 - 3*y + (2.73)**x
grad_f_expr = [ sympy.diff(f_expr, x), sympy.diff(f_expr, y) ]

f_eval = sympy.lambdify((x,y), f_expr, modules='numpy')
grad_f_eval = sympy.lambdify((x,y), grad_f_expr, modules='numpy')

df_dx_expr,df_dy_expr = grad_f_expr
xi = x - alpha*df_dx_expr
etta = y - alpha*df_dy_expr

f_next_prime_expr = sympy.diff(f_expr.subs({x:xi, y:etta}), alpha)
f_next_prime_eval = sympy.lambdify((x,y,alpha), f_next_prime_expr, modules='numpy')

def steepest_descent(gradFunc, p0, f_tolerance=1e-8, maxIter=1000):
    points = [p0]

    for i in range(maxIter):
        x,y = points[i]
        # grad(f) = [2x + y+ e^x; x + 2y - 3]
        # Condition:
        # f'(x_i - alpha_i*grad(f)) = f'(x - alpha*(2x + y + e^x), y - alpha(x + 2y - 3)) =
        # = f'(xi,etta)*etta'*xi' = 2*xi*xi' + xi*etta' + etta*xi' + 2*etta*etaa' - 3*etta' + e^(xi)*xi' =
        # = - 2(x(1-2*alpha) - y - e^x)*alpha*(2x + y + e^x) + ()

        # xi,etta = p - alpha*grad(f)(p)
        # solve f_prime = 0 for alpha
        p = points[i] - alpha_sol*grad_f_eval;
        points.append(p)

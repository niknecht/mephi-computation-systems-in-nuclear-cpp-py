# Agh, still no convergince sufficient conditions!
# !!!this isn't real math!!!

import sympy
import numpy

from scipy.optimize import brentq

x,y,alpha = sympy.symbols('x,y,alpha', real=True)

from math import e as exp

f_expr = exp**(2*(x+y)) + exp**(-x) + exp**(-y)
grad_f_expr = [ sympy.diff(f_expr, x), sympy.diff(f_expr, y) ]

f_eval = sympy.lambdify((x,y), f_expr, modules='numpy')
grad_f_eval = sympy.lambdify((x,y), grad_f_expr, modules='numpy')
df_dx_expr,df_dy_expr = grad_f_expr
xi = x - alpha*df_dx_expr
etta = y - alpha*df_dy_expr

f_next_prime_expr = sympy.diff(f_expr.subs({x:xi, y:etta}, simultaneous=True), alpha)
f_next_prime_eval = sympy.lambdify((x,y,alpha), f_next_prime_expr, modules='numpy')

def errand_grad_descent(p0, step, grad_f_tolerance=1e-8, maxIter=100000):
    points = [p0]

    for i in range(maxIter): #TODO: break cond; find p
        p = numpy.asarray(points[-1], dtype=float)

        p = p - step*numpy.asarray(grad_f_eval(*p));
        points.append(p)

        if(numpy.linalg.norm(numpy.asarray(grad_f_eval(*p))) < grad_f_tolerance):
            break

    return points

def errand_grad_descent_ensure_armijo(p0, step_fyi, grad_f_tolerance=1e-8, maxIter=100000):
    point_steps = [[p0, 0]]

    for i in range(maxIter): #TODO: break cond; find p
        x,y,alpha = sympy.symbols('x,y,alpha', real=True)
        current = numpy.asarray(point_steps[-1][0], dtype=float)

        # step length p_k here (not to be confused with the currect point p) is the negative gradient
        # so the minus begore is calcels out in the xi and etta expressions
        phi_expr = f_expr.subs({x:xi, y:etta}, simultaneous=True)
        phi_eval = sympy.lambdify((x,y,alpha), phi_expr, modules="numpy")
        phi = lambda step: phi_eval(*current, step)
        phi_prime_eval = lambda x,y,step: f_next_prime_eval(x,y,step) # The latter is the same as the former
        phi_prime = lambda step: phi_prime_eval(current[0], current[1], step)

        armijo_c = 1.e-4 # The derivative by alpha at this alpha here is guarranteed by method negative
        hi = 10*step_fyi if step_fyi<.1 else 1# Max step len

        hi_too_hi = lambda hi: phi(hi) <= f_eval(*current) + armijo_c*hi * numpy.dot(numpy.asarray(grad_f_eval(*current)),numpy.asarray(grad_f_eval(*current)))
        while(hi_too_hi(hi)):
            hi*=.5

        alpha = hi

        next = current - alpha*numpy.asarray(grad_f_eval(*current));
        point_steps.append([next, alpha])

        if(numpy.linalg.norm(numpy.asarray(grad_f_eval(*current))) < grad_f_tolerance):
            break

    # Note that curvature is not met, so this still might as well not converge
    return point_steps # **

def errand_grad_descent_ensure_curvature(p0, step_fyi, curv_c=1.e-4, grad_f_tolerance=1e-8, maxIter=100000):
    point_steps = [[p0, 0]]

    for i in range(maxIter): #TODO: break cond; find p
        x,y,alpha = sympy.symbols('x,y,alpha', real=True)
        current = numpy.asarray(point_steps[-1][0], dtype=float)

        # step length p_k here (not to be confused with the currect point p) is the negative gradient
        # so the minus begore is calcels out in the xi and etta expressions
        phi_expr = f_expr.subs({x:xi, y:etta}, simultaneous=True)
        phi_eval = sympy.lambdify((x,y,alpha), phi_expr, modules="numpy")
        phi = lambda step: phi_eval(*current, step) # Homie-phi bro
        phi_prime_eval = lambda x,y,step: f_next_prime_eval(x,y,step) # The latter is the same as the former
        phi_prime = lambda step: phi_prime_eval(current[0], current[1], step)
        grad_phi_expr = [sympy.diff(phi_expr, x), numpy.diff(phi_expr, y)]
        grad_phi_eval = sympy.lambdify((x,y,alpha), grad_phi_expr)

        lo = step_fyi if step_fyi<1 else 1# Min step len

        lo_too_lo = lambda lo: - numpy.asarray(grad_phi_eval(*current, lo))*numpy.asarray(grad_f_eval(*current)) >= - curv_c * numpy.linalg.norm(grad_f_eval(*current))**2
        while(lo_too_lo):
            lo*=2.0

        alpha = lo

        # Really should be steepest_descent_alpha_eval_done instead of alpha
        next = current - alpha*numpy.asarray(grad_f_eval(*current));
        point_steps.append([next, alpha])

        if(numpy.linalg.norm(numpy.asarray(grad_f_eval(*current))) < grad_f_tolerance):
            break
    # Note that Armijo is not met, so this still might as well not converge
    return point_steps # **
# ------------------------- Actual programm -----------------------------
from math import inf
plain_errand_data:list[list[float]] = []
opt_step_iterations:float = +inf
opt_step:float = 0
how_fast:int = 0
#for p in range(-int(4), int(0)):
#    for m_e_pos1 in range(10,99):
#        from math import log,e
#        step:float = m_e_pos1*10**(p-1)
#        iterations = errand_grad_descent((-2,4), step) # TODO: Also change the funciton to the correct one
#        if(len(iterations)<100000):
#            plain_errand_data.append([step, len(iterations)])
#            if(len(iterations)<opt_step_iterations):
#                opt_step = step
opt_step = 0.35

plain_errand_data.sort(key = lambda p: p[0])

from matplotlib import pyplot
fig, (ax1,ax2) = pyplot.subplots(1,2)
ax1.plot([p[0] for p in plain_errand_data], [p[1] for p in plain_errand_data], "-",
         label="''Gradient descent'' (not globally convergent): iterations (the ones that do converge) against step length.")

point_step = errand_grad_descent_ensure_armijo((-2,4), opt_step*10)
ax2.plot([float(i) for i,p_s in enumerate(point_step)], [p_s[1] for p_s in point_step], label="lalala", color="red")

point_step = errand_grad_descent_ensure_curvature((-2,4), opt_step)
ax2.plot([float(i) for i,p_s in enumerate(point_step)], [p_s[1] for p_s in point_step], label="lalala", color="green")


pyplot.show()
pyplot.legend()

# * These lectures should really be more in-depth. We can handle the heat.
# ** Ofc the convergince requires Wolfe be satisfied as well as certain consitions on f
# from which Zoutendijk condition follows. Then from Zoutendijk condition convergence
# follows only for a certain class of algorithms. Thus none of these algorithms nessesarily
# converge globally.
# *** Knowing Wolfe conditions without knowing the above remarks is useless, and the confusion
# this knowledge brings is a disservice. But that's just imho, not to be taken personally.

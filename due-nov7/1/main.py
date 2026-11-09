import sympy
import numpy

from scipy.optimize import brentq

x,y,alpha = sympy.symbols('x,y,alpha', real=True)

f_expr = sympy.exp(2*(x+y)) + sympy.exp(-x) + sympy.exp(-y)
grad_f_expr = [ sympy.diff(f_expr, x), sympy.diff(f_expr, y) ]

f_eval = sympy.lambdify((x,y), f_expr, modules='numpy')
grad_f_eval = sympy.lambdify((x,y), grad_f_expr, modules='numpy')
df_dx_expr,df_dy_expr = grad_f_expr
xi = x - alpha*df_dx_expr
etta = y - alpha*df_dy_expr

f_next_prime_expr = sympy.diff(f_expr.subs({x:xi, y:etta}, simultaneous=True), alpha)
f_next_prime_eval = sympy.lambdify((x,y,alpha), f_next_prime_expr, modules='numpy')

def errand_grad_descent(p0, step, grad_f_tolerance=1e-6, maxIter=100000):
    points = [p0]

    for i in range(maxIter): #TODO: break cond; find p
        p = numpy.asarray(points[-1], dtype=float)

        p = p - step*numpy.asarray(grad_f_eval(*p));
        points.append(p)

        if(numpy.linalg.norm(numpy.asarray(grad_f_eval(*p))) < grad_f_tolerance):
            break

    return points


phi_expr = f_expr.subs({x:xi, y:etta}, simultaneous=True)
phi_eval = sympy.lambdify((x,y,alpha), phi_expr, modules="numpy")
phi_prime_eval = lambda x,y,step: f_next_prime_eval(x,y,step) # The latter is the same as the former
grad_phi_expr = [sympy.diff(phi_expr, x), sympy.diff(phi_expr, y)]
grad_phi_eval = sympy.lambdify((x,y,alpha), grad_phi_expr)

def errand_grad_descent_ensure_armijo(step_fyi, p0=[-2,4],grad_f_tolerance=1e-6, maxIter=100000):
    point_steps = [[numpy.asarray(p0, dtype=float), 0.0]]
    c1 = 1e-4

    for _ in range(maxIter):
        current = point_steps[-1][0]
        g0 = numpy.asarray(grad_f_eval(*current), dtype=float)
        if numpy.linalg.norm(g0) < grad_f_tolerance:
            break

        alpha = step_fyi
        f0 = f_eval(*current)
        g0_sq = numpy.dot(g0, g0)

        while not f_eval(*(current - alpha * g0)) <= f0 - c1 * alpha * g0_sq:
            alpha *= 0.5

        next_p = current - alpha * g0
        point_steps.append([next_p, alpha])

    return point_steps

def errand_grad_descent_ensure_curvature(step_fyi, p0=[-2,4], c2=0.9, grad_f_tolerance=1e-6, maxIter=100000):
    point_steps = [[numpy.asarray(p0, dtype=float), 0.0]]

    for _ in range(maxIter):
        current = point_steps[-1][0]
        g0 = numpy.asarray(grad_f_eval(*current), dtype=float)

        if numpy.linalg.norm(g0) < grad_f_tolerance:
            break

        alpha = step_fyi
        phi_prime0 = -numpy.dot(g0, g0)

        while not (-numpy.dot(numpy.asarray(grad_f_eval(*(current - alpha * g0)), dtype=float), g0) >= c2 * phi_prime0):
            alpha *= 2.0

        next_p = current - alpha * g0
        point_steps.append([next_p, alpha])

    return point_steps
## ------------------------- Actual programm -----------------------------
from math import inf
plain_errand_data:list[list[float]] = []
opt_step_iterations:float = +inf
opt_step:float = 0
how_fast:int = 0
for p in range(-int(4), int(0)):
    for m_e_pos1 in range(10,99):
        from math import log,e
        step:float = m_e_pos1*10**(p-1)
        iterations = errand_grad_descent((-2,4), step) # TODO: Also change the funciton to the correct one
        if(len(iterations)>=100000):
            break
        else:
            plain_errand_data.append([step, len(iterations)])
            if(len(iterations)<opt_step_iterations):
                opt_step = step

plain_errand_data.sort(key = lambda p: p[0])

from matplotlib import pyplot
fig, (ax1,ax2) = pyplot.subplots(1,2)
ax1.plot([p[0] for p in plain_errand_data], [p[1] for p in plain_errand_data], "-",
         label="''Gradient descent'' (not globally convergent): iterations (the ones that do converge) against step length.")

point_step = errand_grad_descent_ensure_armijo(opt_step*10)
ax2.plot([float(i) for i,p_s in enumerate(point_step)], [p_s[1] for p_s in point_step], "-", label="lalala", color="red")

point_step = errand_grad_descent_ensure_curvature(opt_step/10) # Sorry, if I start with opt_step, I get garbage
# because the optimal step also happens to be the largest where this thing still converges
ax2.plot([float(i) for i,p_s in enumerate(point_step)], [p_s[1] for p_s in point_step], "-", label="lalala", color="green")


# The AI written programm actually shown almost the exact same result, so I guess I can call it a day
# Good enough for government work I've just spent 10 hours on this thingy I'm not spending a minute more idc

point_step = errand_grad_descent_ensure_curvature(opt_step/10, c2=0.5) # Sorry, if I start with opt_step, I get garbage
# because the optimal step also happens to be the largest where this thing still converges
ax2.plot([float(i) for i,p_s in enumerate(point_step)], [p_s[1] for p_s in point_step], "-", label="lalala", color="orange")

point_step = errand_grad_descent_ensure_curvature(opt_step/10, c2=0.1) # Sorry, if I start with opt_step, I get garbage
# because the optimal step also happens to be the largest where this thing still converges
ax2.plot([float(i) for i,p_s in enumerate(point_step)], [p_s[1] for p_s in point_step], "-", label="lalala", color="yellow")

ax2.axhline(y=opt_step, linestyle="dotted", color="black")
ax1.axvline(x=opt_step, linestyle="dotted", color="black")
pyplot.show()

# Jorge Nocedal, Stephen J. Wright's Numerical Optimization (2nd edition)
# doesn't mention these methods, only the Newton's method, which is completely different

# So

import sympy
import numpy

from scipy.optimize import brentq

x,y,z,alpha = sympy.symbols('x,y,z,alpha', real=True)

def steepest_descent(f_expr, p0, alpha=0.005, grad_f_tolerance=1e-8, maxIter=int(1e5)):
    grad_f_expr = [ sympy.diff(f_expr, x), sympy.diff(f_expr, y), sympy.diff(f_expr, z) ]

    f_eval = sympy.lambdify((x,y,z), f_expr, modules='numpy')
    grad_f_eval = sympy.lambdify((x,y,z), grad_f_expr, modules='numpy')

    df_dx_expr,df_dy_expr,df_dz_expr = grad_f_expr
    xi = x - alpha*df_dx_expr
    etta = y - alpha*df_dy_expr
    zeta = z - alpha*df_dz_expr

    points = [p0]

    for i in range(maxIter):
        p = numpy.asarray(points[-1], dtype=float)

        # Now find alpha that makes f next prime zero under these xi and etta of x, y and alpha
        # Prepping parameters for the func that searches for the optimal parameter value
        steepest_descent_alpha_eval_done = alpha

        p = p - steepest_descent_alpha_eval_done*numpy.asarray(grad_f_eval(*p));
        points.append(p)

        if(numpy.linalg.norm(numpy.asarray(grad_f_eval(*p))) < grad_f_tolerance):
            break

    return [points[-2], points[-1]]

# So we're just using methods without proofs of global convergence huh

def who_this_iteration_on_that_NAG(f_expr, p0, alpha=0.005, grad_f_tolerance=1e-8, momentum=0.9, maxIter=int(1e5)): # Now I asked you who's this nigga up on that nauauauauaauag?!
    grad_f_expr = [ sympy.diff(f_expr, x), sympy.diff(f_expr, y), sympy.diff(f_expr, z) ]

    f_eval = sympy.lambdify((x,y,z), f_expr, modules='numpy')
    grad_f_eval = sympy.lambdify((x,y,z), grad_f_expr, modules='numpy')

    df_dx_expr,df_dy_expr,df_dz_expr = grad_f_expr
    xi = x - alpha*df_dx_expr
    etta = y - alpha*df_dy_expr

    # Useful definitions

    points = [numpy.asarray(p0, dtype=float)]
    previous = points[0].copy()

    for i in range(maxIter):
        current = points.copy()[-1]
        lookahead = current + momentum * (current - previous)

        grad = numpy.asarray(grad_f_eval(*lookahead), dtype=float).reshape(-1)

        p = lookahead - alpha * grad

        points.append(p.copy())
        previous = current

        if numpy.linalg.norm(
            numpy.asarray(grad_f_eval(*p), dtype=float).reshape(-1)
        ) < grad_f_tolerance:
            break

        if(numpy.linalg.norm(numpy.asarray(grad_f_eval(*p))) < grad_f_tolerance):
            break
    return [points[-2], points[-1]]

# ------------------------- Actual program -----------------------------

# All stationary y follow:
# y = 0.51x^2 + 0.075.
# f(x)=  x^6  − 2.44*x^4 + 1.3354x^2 − 0.26^x − 0.01125
# f'(x)= 6x^5 − 9.76*x^3 + 2.6708x   − 0.26 = 0
# That would give us to 5 stationary x points
# Since d^2/dy^2(f) = 4 > 0 => Hessian is not positieve definite
# And the function f(x, y) cannot have any maxima
#

import csv

allCoord = []

# !TODO: This' gotta go
with open('data_04.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader, None)
    for row in csv_reader:

        allCoord.append([float(row[0]), float(row[1])])

n = len(allCoord)
a0,b0,r0 = (0.0,0.0,0.0)
_f_expr = sympy.Integer(0)
for p in allCoord:
    x_i,y_i = p
    # x is a, y is b, and z is r
    _f_expr = _f_expr + (((x_i - x)**2 + (y_i - y)**2)**0.5-z)**2

    a0+=x_i
    b0+=y_i
a0/=n
b0/=n

for p in allCoord:
    r0+=numpy.linalg.norm(numpy.array([a0,b0])-numpy.asarray(p))
r0/=n

a_sd,b_sd,r_sd = steepest_descent(_f_expr.copy(), (a0,b0,r0))[-1]
a_nag,b_nag,r_nag = who_this_iteration_on_that_NAG(_f_expr.copy(), (a0,b0,r0))[-1]
# From here on out x is x, y is y, and z is z.

import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

_f_nag_expr = (((x - a_nag)**2 + (y - b_nag)**2)**0.5-r_nag)**2
_f_sd_expr = (((x - a_sd)**2 + (y - b_sd)**2)**0.5-r_sd)**2

fig, ax = plt.subplots()
ax.scatter([p[0] for p in allCoord], [p[1] for p in allCoord], color="red", marker=".")

X,Y = numpy.meshgrid(
        numpy.linspace(-8.5, 1.5, 100),
        numpy.linspace(-7, 3, 100))

fig.subplots_adjust(left=0.25)
ax.set_aspect("equal", adjustable="box")
ax.grid(True)

# Just drawing 2 lines here feels a bit worng even though
# we're only interested in the best circle distances from which
# are minimal
# So here's a heatmap for objective function at each center with optinal radii
distances = [
    sympy.sqrt((sympy.sympify(x_i) - x)**2 + (sympy.sympify(y_i) - y)**2)
    for x_i, y_i in allCoord
]

_r_expr = sum(distances, sympy.Integer(0)) / len(distances)

_f_eval = sympy.lambdify((x, y, z), _f_expr, "numpy")
_r_eval = sympy.lambdify((x, y), _r_expr, "numpy")

R_best = _r_eval(X,Y)
J = _f_eval(X,Y,R_best)

heat = ax.contourf(X, Y, J, levels=40, cmap="viridis", alpha=0.75)
fig.colorbar(heat, ax=ax, label="Sum of squared errors")

# Now for the circles
theta = numpy.linspace(0, 2*numpy.pi, 500)
nag_line, = ax.plot(
    a_nag + r_nag*numpy.cos(theta),
    b_nag + r_nag*numpy.sin(theta),
    color="red", label="NAG", zorder=4)
ax.set_xlim(-8.5,1.5)
sd_line, = ax.plot(
    a_sd + r_sd*numpy.cos(theta),
    b_sd + r_sd*numpy.sin(theta),
    color="lime", label="SD", zorder=4)
ax.set_ylim(-7,3)

button_ax = fig.add_axes((0.03, 0.42, 0.18, 0.16))
buttons = CheckButtons(button_ax, ["NAG", "SD"], [True, True])
lines = {"NAG": nag_line, "SD": sd_line}

def this_shoulda_been_lambda_but_this_lang_is_abit_retarded(label):
            line = lines[label]
            line.set_visible(not line.get_visible())
            fig.canvas.draw_idle()

buttons.on_clicked(
        this_shoulda_been_lambda_but_this_lang_is_abit_retarded
)

plt.show()

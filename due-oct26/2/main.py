# Jorge Nocedal, Stephen J. Wright's Numerical Optimization (2nd edition)
# doesn't mention these methods, only the Newton's method, which is completely different

# So

import sympy
import numpy

from scipy.optimize import brentq

x,y,alpha = sympy.symbols('x,y,alpha', real=True)

_f_expr = x**2*(x**2 - 1.22)**2 - 0.26*x + 2*(y - 0.51*x**2)**2 - 0.3*y

def steepest_descent(f_expr, p0, grad_f_tolerance=0.0, maxIter=4000):
    grad_f_expr = [ sympy.diff(f_expr, x), sympy.diff(f_expr, y) ]

    f_eval = sympy.lambdify((x,y), f_expr, modules='numpy')
    grad_f_eval = sympy.lambdify((x,y), grad_f_expr, modules='numpy')

    df_dx_expr,df_dy_expr = grad_f_expr
    xi = x - alpha*df_dx_expr
    etta = y - alpha*df_dy_expr

    f_next_prime_expr = sympy.diff(f_expr.subs({x:xi, y:etta}, simultaneous=True), alpha)
    f_next_prime_eval = sympy.lambdify((x,y,alpha), f_next_prime_expr, modules='numpy')
    # Useful definitions

    points = [p0]

    for i in range(maxIter):
        p = numpy.asarray(points[-1], dtype=float)

        # Now find alpha that makes f next prime zero under these xi and etta of x, y and alpha
        # Prepping parameters for the func that searches for the optimal parameter value
        lo = -1.0 # The derivative by alpha at this alpha here is guarranteed by method negative
        hi = 1.0 # Max step len
        f_next_prime_of_alpha_eval = lambda alpha: float(f_next_prime_eval(*p, alpha))
        while(f_next_prime_of_alpha_eval(hi)*f_next_prime_of_alpha_eval(lo) > 0):
            hi*=2.0
            lo*=2.0
        steepest_descent_alpha_eval_done = brentq(f_next_prime_of_alpha_eval, lo, hi)

        p = p - steepest_descent_alpha_eval_done*numpy.asarray(grad_f_eval(*p));
        points.append(p)

        if(numpy.linalg.norm(numpy.asarray(grad_f_eval(*p))) < grad_f_tolerance):
            break

    return [points[-2], points[-1]]

# So we're just using methods without proofs of global convergence huh

def momentum_descent(f_expr, p0, momentum, maxIter=4000):
    grad_f_expr = [ sympy.diff(f_expr, x), sympy.diff(f_expr, y) ]

    f_eval = sympy.lambdify((x,y), f_expr, modules='numpy')
    grad_f_eval = sympy.lambdify((x,y), grad_f_expr, modules='numpy')

    df_dx_expr,df_dy_expr = grad_f_expr
    xi = x - alpha*df_dx_expr
    etta = y - alpha*df_dy_expr

    f_next_prime_expr = sympy.diff(f_expr.subs({x:xi, y:etta}, simultaneous=True), alpha)
    f_next_prime_eval = sympy.lambdify((x,y,alpha), f_next_prime_expr, modules='numpy')
    # Useful definitions

    points = [p0]

    for i in range(maxIter):
        p = numpy.asarray(points[-1], dtype=float)

        # Now find alpha that makes f next prime zero under these xi and etta of x, y and alpha
        # Prepping parameters for the func that searches for the optimal parameter value
        lo = -1.0 # The derivative by alpha at this alpha here is guarranteed by method negative
        hi = 1.0 # Max step len
        f_next_prime_of_alpha_eval = lambda alpha: float(f_next_prime_eval(*p, alpha))
        while(f_next_prime_of_alpha_eval(hi)*f_next_prime_of_alpha_eval(lo) > 0):
            hi*=2.0
            lo*=2.0
        steepest_descent_alpha_eval_done = brentq(f_next_prime_of_alpha_eval, lo, hi)

        p = p - steepest_descent_alpha_eval_done*numpy.asarray(grad_f_eval(*p)) + momentum*(p-numpy.asarray(points[i-2 if i>2 else -1]));
        points.append(p)
    return [points[-2], points[-1]]

def who_this_iteration_on_that_NAG(f_expr, p0, momentum, maxIter=4000): # Now I asked you who's this nigga up on that nauauauauaauag?!
    grad_f_expr = [ sympy.diff(f_expr, x), sympy.diff(f_expr, y) ]

    f_eval = sympy.lambdify((x,y), f_expr, modules='numpy')
    grad_f_eval = sympy.lambdify((x,y), grad_f_expr, modules='numpy')

    df_dx_expr,df_dy_expr = grad_f_expr
    xi = x - alpha*df_dx_expr
    etta = y - alpha*df_dy_expr

    f_next_prime_expr = sympy.diff(f_expr.subs({x:xi, y:etta}, simultaneous=True), alpha)
    f_next_prime_eval = sympy.lambdify((x,y,alpha), f_next_prime_expr, modules='numpy')
    # Useful definitions

    points = [p0]

    for i in range(maxIter):
        p = numpy.asarray(points[-1].copy(), dtype=float)

        # Now find alpha that makes f next prime zero under these xi and etta of x, y and alpha
        # Prepping parameters for the func that searches for the optimal parameter value
        lo = -1.0 # The derivative by alpha at this alpha here is guarranteed by method negative
        hi = 1.0 # Max step len
        f_next_prime_of_alpha_eval = lambda alpha: float(f_next_prime_eval(*p, alpha))
        while(f_next_prime_of_alpha_eval(hi)*f_next_prime_of_alpha_eval(lo) > 0):
            hi*=2.0
            lo*=2.0

        steepest_descent_alpha_eval_done = brentq(f_next_prime_of_alpha_eval, lo, hi)

        p[1] = p[0] + momentum*(p[0]-numpy.asarray(points[i-2 if i>2 else -1])[0]);
        p[0] = p[1] - steepest_descent_alpha_eval_done*numpy.asarray(grad_f_eval(*p))[1]
        points.append(p)
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

import matplotlib.pyplot as plt

x_grid = numpy.linspace(-1.8, 1.8, 60)
y_grid = numpy.linspace(-1.0, 1.8, 60)
X,Y = numpy.meshgrid(x_grid, y_grid, indexing="xy")

Z = X**2*(X**2 - 1.22)**2 - 0.26*X + 2*(Y - 0.51*X**2)**2 - 0.3*Y

plt.imshow(Z, cmap='viridis', origin='lower')

x,y = sympy.symbols('x,y', real = True)
f_prime_of_x = sympy.diff(_f_expr.subs({y: 0.51*x**2 + 0.075}) ,x)#x**6 - 2.44*x**4 + 1.3354*x**2 - 0.26*x - 0.01125, x)

finish_xs = sympy.solve(f_prime_of_x, x)

finish_points = [
    (float(s[0]), float(s[1]))
    for s in ([r, 0.51 * r**2 + 0.075] for r in finish_xs)
    if numpy.all(
        numpy.linalg.eigvalsh(
            numpy.asarray(sympy.lambdify((x,y), sympy.hessian(_f_expr, (x,y)))(s[0], s[1]), dtype=float)
        ) > 0
    )
]

plt.scatter([p[0] for p in finish_points], [p[1] for p in finish_points], color="green", marker="x")

finish_colors = plt.cm.hsv(
    numpy.linspace(0, 1, len(finish_points), endpoint=False)
)

# Make an exaustive list of colors, assign them to finishes impertively

momentum_stat = 0
nag_stat = 0 # I count 6 points, ...
steep_stat = 0

for ix,x0 in enumerate(x_grid):
    for iy,y0 in enumerate(y_grid):
        start = numpy.asarray([x0,y0])
        nag_minimum_finish = who_this_iteration_on_that_NAG(_f_expr, start.copy(), .1) # I snatch yo black ass off nag down here in the mud!..
        steepest_descent_finish = steepest_descent(_f_expr, start.copy()) # This schozofrenic language passes variables by mutable referenec BY DEFAULT?????????????????
        momentum_descent_finish = momentum_descent(_f_expr, start.copy(), .1)

        # for each of the three check if they're actually close to any one of the actual finishes
        # and what the last step length was to make sure they've actually converged to here
        # color accordingly

        # Exaustive list of possible  colors
        # colors = ["F54927", "27F5B0", "F5B027", "27D3F5", "F5276C", "27F549", "4927F5" ]


        for name, f in [
                ("NAG", nag_minimum_finish),
                ("steep", steepest_descent_finish),
                ("momentum", momentum_descent_finish)
                ]:
            point = numpy.asarray(f[1], dtype=float).reshape(-1)
            nearest = numpy.argmin(
                numpy.linalg.norm(finish_points - point, axis=1)
            )

            cond = abs(point[1] - finish_points[nearest][1]) <= 0.01 and numpy.linalg.norm(f[1]-f[0]) < 0.05
            color = (finish_colors[nearest] if cond else "grey")
            plt.scatter(point[0], point[1],
                        color = color,
                        marker="o")
            if(cond):
                if name == "NAG":
                    nag_stat+=1
                elif name == "momentum":
                    momentum_stat+=1
                elif name == "steep":
                    steep_stat+=1


plt.show()

print(f"Momentum: {momentum_stat/3600*100}%")
print(f"NAG: {nag_stat/3600*100}%")
print(f"Steepest: {steep_stat/3600*100}%")

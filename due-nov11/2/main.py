# The matrix midification algorithm based on modified algorithm Cholesky
import sympy
import numpy
from typing import List,Tuple,Iterable,Callable,Final,cast
from numpy.typing import NDArray

x,y = sympy.symbols("x,y", real=True)

# see "Modified Cholesky algorithm based on a symmetric indefinite factorization", Cheng, Sheung Hun and Higham, Nicholas J. 1998
def mc(current_hessian:NDArray[numpy.float64], delta:numpy.float64)->NDArray[numpy.float64]:
    A:Final = current_hessian;
    from scipy.linalg import ldl;
    lu, D, perm = ldl(A, lower=True, hermitian=True)
    # Hopefully these catch the error, no handling since this should never fail
    # and if it does there is something seriously wrong, a bug
    lu:NDArray[numpy.float64] = numpy.asarray(lu,dtype=numpy.float64);
    D:NDArray[numpy.float64] = numpy.asarray(D,dtype=numpy.float64);
    perm:NDArray[numpy.intp] = numpy.asarray(perm,dtype=numpy.int64);

    L = lu[perm, :]

    D_hat = D.copy()
    i = 0

    while i < D_hat.shape[0]:
        block_size = 2 if i + 1 < D_hat.shape[0] and D_hat[i + 1, i] != 0.0 else 1
        sl = slice(i, i + block_size)

        block = D_hat[sl, sl]
        block = 0.5 * (block + block.T)
        eigvals, Q = numpy.linalg.eigh(block)
        D_hat[sl, sl] = (Q * numpy.maximum(eigvals, delta)) @ Q.T

        i += block_size

    A_mod_perm = L @ D_hat @ L.T

    A_mod = numpy.empty_like(A)
    A_mod[numpy.ix_(perm, perm)] = A_mod_perm
    A_mod = 0.5 * (A_mod + A_mod.T)

    return A_mod
    # A hessian matrix is symmetric hermitian

    return A;
def newton(f_expr, p0:Tuple[float,float], step_hint:float=0.01, gradTolerance:float=1e-6, maxIter:int=1000) -> List[Tuple[float,float]]:
    grad_f_expr = (sympy.diff(f_expr,x), sympy.diff(f_expr, y))
    grad_f_eval:Callable[[float,float], float] = sympy.lambdify([x,y], grad_f_expr, "numpy")

    f_eval:Callable[[float, float], float] = sympy.lambdify([x,y], f_expr, "numpy")

    hess_f_expr = sympy.hessian(f_expr, (x,y))
    hess_f_eval:Callable[[float,float], float] = sympy.lambdify([x,y], hess_f_expr)

    points:List[Tuple[float,float]] = [p0]
    current:Tuple[float,float] = p0
    next:Tuple[float,float]

        # - Check that Hessian is positive definite and even exists, it must
        # be positive definete in order to satisfy the descent direction requirement
        # - Chech that gradient exists and is good
        # -  line_search for this p if and only if the matrix is positive definite, otherwise
        # keep adding identity metricies to it untill it is
        # if so follow the same logics from the previous lab.

        # - Hessian inverse
        # find the inverse, see if finite, if not


    def f_at(z:Tuple[float,float]):
        return float(f_eval(float(z[0]), float(z[1])))

    def grad_at(z:Tuple[float,float]):
        return numpy.asarray(
            grad_f_eval(float(z[0]), float(z[1])),
            dtype=numpy.float64
        ).reshape(2)

    import scipy.optimize;

    while numpy.linalg.norm(grad_f_eval(*current))>=gradTolerance:
        grad = grad_at(current)

        if not numpy.all(numpy.isfinite(grad)):
            break
        if numpy.linalg.norm(grad) < gradTolerance:
            break

        hess = numpy.asarray(
            hess_f_eval(float(current[0]), float(current[1])),
            dtype=numpy.float64
        )

        if not numpy.all(numpy.isfinite(hess)):
            break

        scale = max(1.0, float(numpy.linalg.norm(hess, ord=numpy.inf)))
        delta = numpy.sqrt(numpy.finfo(numpy.float64).eps) * scale
        hess_pd = mc(hess, numpy.float64(delta))

        try:
            direction = numpy.linalg.solve(hess_pd, -grad)
        except numpy.linalg.LinAlgError:
            break

        if not numpy.all(numpy.isfinite(direction)) or grad @ direction >= 0.0:
            break

        if(len(points) >= maxIter):
            break

        current_array = numpy.asarray(current, dtype=numpy.float64)
        alpha, _, _, _, _, _ = scipy.optimize.line_search(
            f_at,
            grad_at,
            current,
            direction,
            gfk=grad,
            old_fval=f_at(current),
            c1=1e-4,
            c2=0.9
        )

        if alpha is None:
            break

        next_arr = current + alpha * direction
        if not numpy.all(numpy.isfinite(next_arr)):
            break
        next_point: Tuple[float, float] = (
            float((next_arr)[0]),
            float((next_arr)[1])
        )
        if not numpy.all(numpy.isfinite(next_point)):
            break


        current = next_point
        points.append(current)

    return points

points:List[Tuple[float,float]] = (numpy.loadtxt("./2/data_04.csv", delimiter=",", skiprows=1)).tolist()

f_expr = 0

# a->x, b->y
for p in points:
    t_i, y_i = p
    p_i =  1/(1 + sympy.exp(-(x+y*t_i)))
    f_expr -= y_i*sympy.ln(p_i) + (1 - y_i)*sympy.ln(1 - p_i)

x0:Tuple[float,float] = (0.,0.)

from numpy.typing import NDArray
f_eval = sympy.lambdify([x,y], f_expr)
def f_at(r:NDArray[numpy.float64])->float:
    return f_eval(r[0], r[1])
def grad_f_at(r:NDArray[numpy.float64])->float:
    return grad_f_eval(r[0], r[1])

grad_f_expr = [sympy.diff(f_expr,x), sympy.diff(f_expr,y)]
grad_f_eval = sympy.lambdify([x,y], grad_f_expr)

from matplotlib import pyplot
fig, axs = pyplot.subplots(2, 1, sharex=True, sharey=True)

# My newton algorithm converges in 8 steps
xvals:List[int]
yvals:List[float]

newton_points = newton(f_expr, x0, 0.01)

yvals = cast(List[float], [float(numpy.linalg.norm(grad_f_eval(*p)))  for p in newton_points])
xvals = list(range(len(yvals)))
axs[0].plot(xvals,yvals,"-")
axs[0].set_title(f"Naive newton (modified Cholesky): {len(xvals)} steps")


bfgs_points: List[Tuple[float, float]] = [(float(x0[0]), float(x0[1]))]

def record(xk): # The BGFS callback
    bfgs_points.append((float(xk[0]), float(xk[1])))
import scipy
res = scipy.optimize.minimize(f_at, numpy.asarray(x0, dtype=numpy.float64),
                              jac=grad_f_at, method="BFGS", callback=record)
yvals = cast(List[float], [float(numpy.linalg.norm(grad_f_eval(*p))) for p in bfgs_points])
xvals = list(range(len(yvals)))
axs[1].plot(xvals,yvals,"-")
axs[1].set_title(f"Library BFGS: {len(xvals)} steps")


# Most likely probability distribution model under the known conditions according
# to the newton method
a_newton:float
b_newton:float
a_newton, b_newton = newton_points[-1]

# Most likely probability distribution model under the known conditions according
# to the SR1 method
a_bfgs:float
b_bfgs:float
a_bfgs, b_bfgs = res.x

fig, ax = pyplot.subplots()

ts = numpy.array([p[0] for p in points])
ys = numpy.array([p[1] for p in points])
ax.scatter(ts, ys, label="Conditions")

t_grid = numpy.linspace(ts.min(), ts.max(), 300)
most_likely_ps_newton =  1/(1 + numpy.exp(-(a_newton+b_newton*t_grid)))
most_likely_ps_bfgs =  1/(1 + numpy.exp(-(a_bfgs+b_bfgs*t_grid)))
ax.plot(t_grid, most_likely_ps_newton, label="Most plausible according to newton under the known conditions")
ax.plot(t_grid, most_likely_ps_bfgs, label="Most plausible according to bfgs under the known conditions")

# .9 = 1/(1 + exp(-(a+bt)))
# 1/9 = exp(-(a+bt))
# ln(9) = a+bt
# t = (ln(9) - a)/b
t09_newton = (numpy.log(9) - a_newton)/b_newton
t09_bfgs = (numpy.log(9) - a_bfgs)/b_bfgs

ax.axvline(t09_newton, linestyle="--", label=f"Time to 0.9 according to newton: {t09_newton}")
ax.axvline(t09_bfgs, linestyle="--", label=f"Time to 0.9 according to bfgs: {t09_bfgs}")

y_grid = numpy.linspace(0., 1., 300)
ax.plot()
ax.legend()

pyplot.show()

import sympy
import numpy
from typing import List,Tuple,Iterable,Callable,Final,cast
from numpy.typing import NDArray

x,y = sympy.symbols("x,y", real=True)
def errand_grad_descent(f_expr, p0:Tuple[float,float], step:float, grad_tolerance:float=1.e-6, maxIterBeforeDivergent:int=10000) -> List[Tuple[float, float]]:
    grad_f_expr = [sympy.diff(f_expr,x), sympy.diff(f_expr,y)]
    grad_f_eval:Callable[[float,float], Tuple[float,float]] = cast(Callable[[float,float], Tuple[float,float]], sympy.lambdify((x,y), grad_f_expr))

    points:List[Tuple[float,float]] = [p0]
    current:Tuple[float,float] = p0
    while numpy.linalg.norm(grad_f_eval(*current)) >= grad_tolerance:
        next:Tuple[float,float] = (current - step*numpy.array(grad_f_eval(*current), dtype=float)).tolist()
        points.append(next)

        current = points[-1]

        if(
                not numpy.all(numpy.isfinite(grad_f_eval(*current)))
                or
                not numpy.all(numpy.isfinite(next))
        ): return points
        if(len(points)>maxIterBeforeDivergent):
            return points;

    return points

# So the important part is that the convergence of the newton method comes form the same theorem that convergence
# of the steepest decsent and others, so it still relies on the function being well behaved so that from Wolfe follows
# Zoutendijk. Local rate of convergence can and will be quadratic, but I care more about is this thing even going to
# converge globally, that is proven by the same theorems mentioned in the previous labs. This requires the direction to
# be the descent direction, here meaning the matrix is positive devinite.

# The matrix midification algorithm based on modified algorithm Cholesky
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
def newton(f_expr, p0:Tuple[float,float], step_hint:float=0.1, gradTolerance:float=1e-6, maxIter:int=1000) -> List[Tuple[float,float]]:
    grad_f_expr = (sympy.diff(f_expr,x), sympy.diff(f_expr, y))
    grad_f_eval:Callable[[float,float], float] = sympy.lambdify([x,y], grad_f_expr, "numpy")

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

# What do you know, my MC algorithm implementation comes to the rescue again as
# the SR1 approximation does not have to be positive definite, meaning that times
# the gradient is not nessesarily a descent direction, but MC can produce the matrix
# that is.
def sr1(f_expr, p0:Tuple[float,float], gamma:float=1.0, grad_tolerance:float=1e-6, maxIterBeforeDivergent=10000) -> Tuple[List[Tuple[float,float]], NDArray[numpy.float64]]:
    x,y = sympy.symbols("x,y", real=True)
    grad_f_expr = (sympy.diff(f_expr,x), sympy.diff(f_expr, y))
    grad_f_eval:Callable[[float,float], float] = sympy.lambdify([x,y], grad_f_expr, "numpy")

    hess_f_expr = sympy.hessian(f_expr, (x,y))
    hess_f_eval:Callable[[float,float], float] = sympy.lambdify([x,y], hess_f_expr)

    points:List[Tuple[float,float]] = [p0]
    current:Tuple[float,float] = p0
    next:Tuple[float,float]

    from scipy.optimize import line_search

    if gamma <= 0.0:
        raise ValueError("gamma must be positive")

    f_eval = sympy.lambdify([x, y], f_expr, "numpy")

    def grad_f_at(z) -> NDArray[numpy.float64]:
        return numpy.asarray(
            grad_f_eval(float(z[0]), float(z[1])),
            dtype=numpy.float64
        ).reshape(2)

    def f_at(z):
        return f_eval(float(z[0]), float(z[1]))

    points:List[Tuple[float,float]] = [p0]
    current:Tuple[float,float] = p0
    B:NDArray[numpy.float64] = gamma * numpy.eye(2, dtype=numpy.float64)

    while numpy.linalg.norm(grad_f_eval(*current)) >= grad_tolerance:
        grad = grad_f_eval(*current)

        if not numpy.all(numpy.isfinite(grad)):
            break

        # MC makes a copy used for the direction; keep B as the raw SR1 update.
        scale = max(1.0, float(numpy.linalg.norm(B, ord=numpy.inf)))
        delta = numpy.sqrt(numpy.finfo(numpy.float64).eps) * scale
        B_pd = mc(B, numpy.float64(delta))

        direction = -(B_pd @ grad)

        if not numpy.all(numpy.isfinite(direction)) or grad @ direction >= 0.0:
            break

        current_array = numpy.asarray(current, dtype=numpy.float64)
        alpha, _, _, _, _, _ = line_search(
            f_at,
            grad_f_at,
            current_array,
            direction,
            gfk=grad,
            old_fval=f_at(current_array),
            c1=1e-4,
            c2=0.9
        )

        if alpha is None:
            break

        next_array = current_array + float(alpha) * direction
        if not numpy.all(numpy.isfinite(next_array)):
            break

        next_point:Tuple[float,float] = (
            float(next_array[0]),
            float(next_array[1])
        )
        next_grad = grad_f_eval(*next_point)

        grad = numpy.asarray(grad, dtype=numpy.float64).reshape(2)
        next_grad = numpy.asarray(next_grad, dtype=numpy.float64).reshape(2)
        y = next_grad - grad

        if not numpy.all(numpy.isfinite(next_grad)):
            break

        # Inverse-Hessian SR1 update from the lecture.
        s = next_array - current_array
        y = next_grad - grad
        v = s - B @ y
        denominator = float(v @ y)

        if abs(denominator) >= 1e-8 * numpy.linalg.norm(v) * numpy.linalg.norm(y):
            B = B + numpy.outer(v, v) / denominator
            B = 0.5 * (B + B.T)

        current = next_point
        points.append(current)

        if len(points) > maxIterBeforeDivergent:
            return points, B

    return points, B

f_expr = sympy.exp(2*(x+y)) + sympy.exp(-x) + sympy.exp(-y)
f_eval:Callable[[float,float], float] = sympy.lambdify((x,y), f_expr, "numpy")
x0:Final[Tuple[float,float]] = (-2,4) # -2;12
opt_step:Final[float] = .01


from matplotlib import pyplot
fig, axs = pyplot.subplots(2, 2, sharex=True, sharey=True)

# Same number of steps as before, more that a 100
xvals:List[int]
yvals:List[float]
yvals = cast(List[float], [f_eval(*p) for p in errand_grad_descent(f_expr, x0, opt_step)])
xvals = list(range(len(yvals)))
axs[0,0].plot(xvals,yvals,"-")
axs[0,0].set_title(f"Const step line descent: {len(xvals)} steps")

# My newton algorithm converges in 8 steps
xvals:List[int]
yvals:List[float]
yvals = cast(List[float], [f_eval(*p) for p in newton(f_expr, x0, opt_step)])
xvals = list(range(len(yvals)))
axs[0,1].plot(xvals,yvals,"-")
axs[0,1].set_title(f"Naive newton: {len(xvals)} steps")


sr1_points,B_final = sr1(f_expr, x0)
hess_eval = sympy.lambdify([x, y], sympy.hessian(f_expr, (x, y)), "numpy")
H_star = numpy.asarray(hess_eval(*sr1_points[-1]), dtype=numpy.float64)

# Solve H_star @ X = I rather than explicitly calling inverse().
H_inv_star = numpy.linalg.solve(H_star, numpy.eye(2))

absolute_error = numpy.linalg.norm(B_final - H_inv_star, ord="fro")
relative_error = absolute_error / numpy.linalg.norm(H_inv_star, ord="fro")

print("Absolute Frobenius error:", absolute_error)
print("Relative Frobenius error:", relative_error)
xvals:List[int]
yvals:List[float]
yvals = cast(List[float], [f_eval(*p) for p in sr1(f_expr, x0)[0]])
xvals = list(range(len(yvals)))
axs[1,0].plot(xvals,yvals,"-")
axs[1,0].set_title(f"SR1: {len(xvals)} steps")

pyplot.show()

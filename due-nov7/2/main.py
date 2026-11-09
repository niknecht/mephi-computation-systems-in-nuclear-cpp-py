import sympy
import numpy
from typing import List,Tuple,Iterable,Callable,Final,cast

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

def steepest_descent(f_expr, p0:Tuple[float,float], step_hint:float, grad_tolerance:float=1.e-6, maxIterBeforeDivergent:int=10000) -> List[Tuple[float, float]]:
    from scipy.optimize import line_search
    points:List[Tuple[float,float]]
    grad_f_expr = [sympy.diff(f_expr,x), sympy.diff(f_expr,y)]
    grad_f_eval:Callable[[float,float], Tuple[float,float]] = cast(Callable[[float,float], Tuple[float,float]], sympy.lambdify((x,y), grad_f_expr))

    points:List[Tuple[float,float]] = [p0]
    current:Tuple[float,float] = p0
    while numpy.linalg.norm(grad_f_eval(*current)) >= grad_tolerance:
        step, *_ = line_search(
                lambda p_t: float(f_eval(*p_t)),
                lambda p_t: numpy.asarray(grad_f_eval(*p_t),dtype=float),
                current,
                -numpy.array(grad_f_eval(*current),dtype=float),
                gfk=numpy.array(grad_f_eval(*current),dtype=float)
            )
        if(step is None):
            step = step_hint

        next:Tuple[float,float] = (current - step*numpy.array(grad_f_eval(*current), dtype=float)).tolist()
        points.append(next)

        current = points[-1]

        if(
                not numpy.all(numpy.isfinite(grad_f_eval(*current)))
                or
                not numpy.all(numpy.isfinite(next))
        ):
            return points;
        if(len(points)>maxIterBeforeDivergent):
            return points;

    return points

def errand_nag_search(f_expr, p0:Tuple[float,float], step:float, momentum:float=0.9, grad_tolerance:float=1.e-6, maxIterBeforeDivergent:int=10000) -> List[Tuple[float,float]]:
    points:List[Tuple[float,float]]
    grad_f_expr = [sympy.diff(f_expr,x), sympy.diff(f_expr,y)]
    grad_f_eval:Callable[[float,float], Tuple[float,float]] = cast(Callable[[float,float], Tuple[float,float]], sympy.lambdify((x,y), grad_f_expr))

    points:List[Tuple[float,float]] = [p0]
    current:Tuple[float,float] = p0
    V_vec:Tuple[float,float] = (0.,0.)
    while numpy.linalg.norm(grad_f_eval(*current)) >= grad_tolerance:
        next:Tuple[float,float]

        # This language is terrible, and I'm trying to limit fail scinarios as much as possible
        next:Tuple[float,float] = (current - step*numpy.array(grad_f_eval(*(current + step*numpy.array(V_vec,dtype=float)))
                                                              + step*numpy.array(V_vec,dtype=float),dtype=float)).tolist()
        points.append(next)

        current = points[-1]
        V_vec = (numpy.asarray(current) -numpy.asarray(points[-2])).tolist() # ramaswami

        if(
                not numpy.all(numpy.isfinite(grad_f_eval(*current)))
                or
                not numpy.all(numpy.isfinite(next))
        ):
            return points;
        if(len(points)>maxIterBeforeDivergent):
            return points;

    return points

def nag_search(f_expr, p0:Tuple[float,float], step_hint:float, grad_tolerance:float=1.e-6, momentum=.1, maxIterBeforeDivergent:int=10000) -> List[Tuple[float, float]]:
    grad_f_expr = [sympy.diff(f_expr,x), sympy.diff(f_expr,y)]
    grad_f_eval:Callable[[float,float], Tuple[float,float]] = cast(Callable[[float,float], Tuple[float,float]], sympy.lambdify((x,y), grad_f_expr))
    f_eval:Callable[[float,float], float] = sympy.lambdify((x,y), f_expr)

    V_vec:Tuple[float,float] = (0.,0.)
    points:List[Tuple[float,float]] = [p0]
    current:Tuple[float,float] = p0

    while numpy.linalg.norm(grad_f_eval(*current)) >= grad_tolerance:
        lookahead:Tuple[float,float] = (current + momentum * numpy.array(V_vec,dtype=float)).tolist()
        gradient:Tuple[float,float] = grad_f_eval(*lookahead)
        from scipy.optimize import line_search

        step, *_ = line_search(
                lambda p_t: float(f_eval(*p_t)),
                lambda p_t: numpy.asarray(grad_f_eval(*p_t),dtype=float),
                current,
                -numpy.array(grad_f_eval(*(current+momentum*numpy.array(V_vec,dtype=float))),dtype=float),
                gfk=numpy.array(grad_f_eval(*(current)),dtype=float)
            )
        if step is None:
            step = step_hint

        next:Tuple[float,float]

        # This language is terrible, and I'm trying to limit fail scinarios as much as possible
        next:Tuple[float,float] = (lookahead - step * numpy.array(gradient,dtype=float)).tolist()
            #(current - step*numpy.array(grad_f_eval(*(current + step*numpy.array(V_vec,dtype=float)))
             #                                                 + step*numpy.array(V_vec,dtype=float),dtype=float)).tolist()
        points.append(next)

        current = points[-1]
        V_vec = (numpy.asarray(current) -numpy.asarray(points[-2])).tolist() # ramaswami

        if(
                not numpy.all(numpy.isfinite(grad_f_eval(*current)))
                or
                not numpy.all(numpy.isfinite(next))
        ):
            return points;
        if(len(points)>maxIterBeforeDivergent):
            return points;

    return points


f_expr = sympy.Integer(0)
eps = 0.01
for (xi,yi,mi) in cast(Iterable[Tuple[float,float,float]], [(7.,3.,3.), (10.,2.,4.), (10.,9.,2.), (10.,8.,2.)]):
    f_expr += mi*sympy.sqrt((x - xi)**2 + (y - yi)**2 + eps**2)
f_eval:Callable[[float,float], float] = sympy.lambdify((x,y), f_expr)

x0:Final[Tuple[float,float]] = (-2,12)

iter_step:List[Tuple[int, float]] = []

opt_step:float = 0
from math import inf
opt_iter:int = 100001
for p in range(-5, 0):
    step:float = 10**(p)
    points:List[Tuple[float,float]] = errand_grad_descent(f_expr, x0, step)
    if(len(points) < opt_iter
       and
       numpy.all(numpy.isfinite(points[-1]))):
         opt_step = step
         opt_iter = len(points)

print("And our winner is (km):")
print(opt_step)
print(f"in {opt_iter} steps")

points:List[Tuple[float, float]] = steepest_descent(f_expr, x0, opt_step)
opt_x,opt_y = points[-1]
print(f"Steepest descent: {len(points)} iterations")

from matplotlib import pyplot
fig, axs = pyplot.subplots(2, 2, sharex=True, sharey=True)

f_eval:Callable[[float,float], float] = sympy.lambdify((x,y), f_expr, "numpy")
def ys(points:List[Tuple[float,float]])->List[float]:
    return [f_eval(*_) for _ in points]

xvals:List[int]
yvals:List[float]
yvals = cast(List[float], [f_eval(*p) for p in errand_grad_descent(f_expr, x0, opt_step)])
xvals = list(range(len(yvals)))
axs[0,0].plot(xvals,yvals,"-")
axs[0,0].set_title("Const step line descent")

xvals:List[int]
yvals:List[float]
yvals = cast(List[float], [f_eval(*p) for p in errand_nag_search(f_expr, x0, opt_step)])
xvals = list(range(len(yvals)))
axs[1,0].plot(xvals,yvals,"-")
axs[1,0].set_title("Const step NAG")

xvals:List[int]
yvals:List[float]
yvals = cast(List[float], [f_eval(*p) for p in steepest_descent(f_expr, x0, opt_step)])
xvals = list(range(len(yvals)))
axs[0,1].plot(xvals,yvals,"-")
axs[0,1].set_title("Line search step steepest descent")

xvals:List[int]
yvals:List[float]
yvals = cast(List[float], [f_eval(*p) for p in nag_search(f_expr, x0, opt_step)])
xvals = list(range(len(yvals)))
axs[1,1].plot(xvals,yvals,"-")
axs[1,1].set_title("Line search NAG")

pyplot.show()

import sympy
import numpy
import scipy

# Optimization labs to hotline miami soundtrack at 5 am hit differently
from typing import List,Final
from numpy.typing import NDArray
Ts:Final[NDArray[numpy.float64]] = numpy.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
Rs:Final[NDArray[numpy.float64]] = numpy.array([102.3, 106.7, 111.1, 115.0, 119.4, 123.6, 127.6, 131.9, 136.1, 140.5])

#sigmas:Final[List[float]] = [r-R0*(1+alpha*t) for t,r in zip(Ts,Rs)]

y:NDArray[numpy.float64] = numpy.array([[r] for r in Rs])
#yes matrix yes that makes a difference because this would be an algebra
A:NDArray[numpy.float64] = numpy.array([[1, t] for t in Ts])
# These are dimentia-on tested valid
#u = y-A@alpha -- This doesn't cause an error

# Now assume that the gradient by alpha is zero
# Form Ax = b
vec_alpha = numpy.linalg.solve(numpy.transpose(A)@A, numpy.transpose(A)@y) # [R0, alpha]

vec_alpha_lib,_,_,_ = numpy.linalg.lstsq(A, y) # [R0, alpha]

import matplotlib.pyplot
ax = matplotlib.pyplot.subplot()
ax.scatter(Ts, Rs, label="experimental points")

t_domain = numpy.linspace(min(Ts), max(Ts))
least_sq_approximation = vec_alpha[0] + vec_alpha[1]*t_domain
lib_least_sq_approximation = vec_alpha_lib[0] + vec_alpha_lib[1]*t_domain

ax.plot(t_domain, least_sq_approximation,label=f"my implementation with numpy and analytical sol R0={vec_alpha[0]},\n\
alpha={vec_alpha[1]/vec_alpha[0]}, T(166)={(166.-vec_alpha[0])/vec_alpha[1]}")
ax.plot(t_domain, lib_least_sq_approximation, label=f"library least squares method R0={vec_alpha_lib[0]},\n\
alpha={vec_alpha_lib[1]/vec_alpha_lib[0]}, T(166)={(166.-vec_alpha_lib[0])/vec_alpha_lib[1]}")

ax.legend()
matplotlib.pyplot.show()
# R = R0*(1 + alpha*T)

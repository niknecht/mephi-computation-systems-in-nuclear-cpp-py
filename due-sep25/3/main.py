# Var 4: 2x + 3y, subject to: x^2/9 + y^2/4 - 1 == 0

import sympy
from sympy import Matrix

x,y = sympy.symbols('x,y', real = True)
f = 2*x + 3*y
g = x**2/9 + y**2/4 - 1

gradf = [sympy.diff(f, x), sympy.diff(f, y)]
gradg = [sympy.diff(g, x), sympy.diff(g, y)]

lam = sympy.symbols('lambda')
first,second = [ gradf[0] - lam*gradg[0], gradf[1] - lam*gradg[1] ]
third = x**2/9 + y**2/4 - 1

sol = sympy.solve([first, second, third], [x, y, lam])

L = f - lam * g
H_L = sympy.hessian(L, (x, y))
H_f = Matrix([[0,         sympy.diff(g, x), sympy.diff(g, y)],
              [sympy.diff(g, x), H_L[0, 0], H_L[0, 1]],
              [sympy.diff(g, y), H_L[1, 0], H_L[1, 1]]])

for s in sol:
    if H_f.subs({x: s[0], y: s[1], lam: s[2]}).det() > 0:
        print(f'f({float(s[0])}, {float(s[1])}) = {f.subs({x: float(s[0]), y: float(s[1])})} - max')
    elif H_f.subs({x: s[0], y: s[1], lam: s[2]}).det() < 0:
        print(f'f({float(s[0])}, {float(s[1])}) = {f.subs({x: float(s[0]), y: float(s[1])})} - min')

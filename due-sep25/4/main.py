# Obj. function: a*b*c
# Subject to:
#   x^2/9 + y^2/4 + z^2 - 1 = 0
#   Where x = a/2, y = b/2, z = c/2
#   The rest of the verticies are ommited because symmetry

import sympy
from sympy import Matrix

a,b,c = sympy.symbols('a,b,c', real = True, positive = True)
f = a*b*c
x,y,z = [a/2, b/2, c/2]
g = x**2/9 + y**2/4 + z**2 - 1

gradf = [sympy.diff(f, a), sympy.diff(f, b), sympy.diff(f, c)]
gradg = [sympy.diff(g, a), sympy.diff(g, b), sympy.diff(g, c)]
lam = sympy.symbols('lambda', real = True)

#   std::views::iota(2)
# | std::views::transform([](const auto& i){ return ...; })
# | std::ranges::to<std::vector>();
first,second,third = list(map(lambda i:
                                gradf[i] - lam * gradg[i]
                              ,range(3)))
fourth = g
pot = sympy.solve([first, second, third, fourth], [a, b, c, lam]) # pol

L = f - lam * g
H_L = sympy.hessian(L, [a, b, c])
H_f = Matrix([[0, sympy.diff(g, a), sympy.diff(g, b), sympy.diff(g, c)],
              [sympy.diff(g,a), H_L[0,0], H_L[0, 1], H_L[0, 2]],
              [sympy.diff(g,b), H_L[1,0], H_L[1, 1], H_L[1, 2]],
              [sympy.diff(g,c), H_L[2,0], H_L[2, 1], H_L[2, 2]]])

# C++ ranges are simply superior
viable = list(map(lambda p: {a: p[0], b: p[1], c: p[2]},
                  filter(lambda p: H_f.subs({a: p[0], b: p[1], c: p[2], lam: p[3]}).det() < 0,
                         pot)))

print(viable)

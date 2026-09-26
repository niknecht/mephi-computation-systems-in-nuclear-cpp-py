# Obj. function: a*b*c
# Subject to:
#   x^2/9 + y^2/4 + z^2 - 1 = 0
#   Where x = a/2, y = b/2, z = c/2
#   The rest of the verticies are ommited because symmetry

import sympy
from sympy import Matrix
from functools import reduce

w1,w2,w3 = sympy.symbols('w1,w2,w3', real = True, positive = True)
f = w1*w2*w3
g = w1**2 + 2*w2**2 + 3*w3**2 - 8

gradf = [sympy.diff(f, w1), sympy.diff(f, w2), sympy.diff(f, w3)]
gradg = [sympy.diff(g, w1), sympy.diff(g, w2), sympy.diff(g, w3)]
lam = sympy.symbols('lambda', real = True)

#   std::views::iota(2)
# | std::views::transform([](const auto& i){ return ...; })
# | std::ranges::to<std::vector>();
first,second,third = list(map(lambda i:
                                gradf[i] - lam * gradg[i]
                              ,range(3)))
fourth = g
pot = sympy.solve([first, second, third, fourth], [w1, w2, w3, lam]) # pol

L = f - lam * g
H_L = sympy.hessian(L, [w1, w2, w3])
H_f = Matrix([[0, sympy.diff(g, w1), sympy.diff(g, w2), sympy.diff(g, w3)],
              [sympy.diff(g,w1), H_L[0,0], H_L[0, 1], H_L[0, 2]],
              [sympy.diff(g,w2), H_L[1,0], H_L[1, 1], H_L[1, 2]],
              [sympy.diff(g,w3), H_L[2,0], H_L[2, 1], H_L[2, 2]]])

# C++ ranges are simply superior
viable = list(map(lambda p: {w1: p[0], w2: p[1], w3: p[2]},
                  filter(lambda p: H_f.subs({w1: p[0], w2: p[1], w3: p[2], lam: p[3]}).det() < 0,
                         pot)))

# WHY do I have to go in reverse over operations???
# WHY can't this language be normal here?
mods = list(map(lambda point:
                    sympy.sqrt(reduce(lambda result, current:
                            result + current
                           , list(map(lambda coor: coor**2, point.values()))))
                  , viable))

# I HATE HATE HATE this abomination of a language
print(list(map(lambda point, mod:
                list(map(lambda coor: float(coor/mod), point.values()))
               , viable, mods)))

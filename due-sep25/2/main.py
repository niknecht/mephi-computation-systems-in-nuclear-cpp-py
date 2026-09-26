import sympy

# Var 4: x^2 + y^2 + cosh(x - y)

x,y = sympy.symbols('x,y', real = True)
f = x**2 + y**2 + sympy.cosh(x - y)

# gradf = [sympy.diff(f, x), sympy.diff(f, y)]
# ext = sympy.solve(gradf, [x, y], dict=True) # sympy can't solve this, therefore the only solution with python is a graphical one

#sympy.plotting.plot3d(f, (x, -1,  1), (y, -1,  1))
ext = [{x: 0, y: 0}]

plusDelta = sympy.Dummy('x,y', positive = True)
viable = list(filter(lambda pt:
                     sympy.hessian(f, [x, y]).subs(pt).is_positive_definite
                     or
                     sympy.hessian(f, [x, y]).subs(pt).is_negative_definite
                 ,ext))
print(list(zip(
    viable,
    list(map(lambda pt:
                    f.subs(pt)
                ,viable)))))

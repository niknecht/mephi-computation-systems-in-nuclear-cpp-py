import sympy

# Var 4: x + 1/x

x = sympy.symbols('x', real = True)
f = x + 1/x
# First derivative is enough for this variant
dfdx = sympy.diff(f, x)
ext = sympy.solve(dfdx, x)

plusDeltaX = sympy.Dummy('x', positive = True)
viable = list(filter(lambda pt:
                     sympy.sign(dfdx.subs(x, pt + plusDeltaX).as_leading_term(plusDeltaX).as_coeff_exponent(plusDeltaX)[0]) == 1
                     and
                     sympy.sign(dfdx.subs(x, pt - plusDeltaX).as_leading_term(plusDeltaX).as_coeff_exponent(plusDeltaX)[0]) == -1
                 ,ext))

print(list(zip(
    viable,
    list(map(lambda pt:
                    f.subs(x, pt)
                ,viable)))))

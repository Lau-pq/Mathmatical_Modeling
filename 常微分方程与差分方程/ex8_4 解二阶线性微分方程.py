import sympy as sp

x = sp.var('x'); y = sp.Function('y')
eq = y(x).diff(x, 2) - 2 * y(x).diff(x) + y(x) - sp.exp(x)
con = {y(0) : 1, y(x).diff(x).subs(x, 0) : -1}
s = sp.dsolve(eq, ics=con)
print(s)
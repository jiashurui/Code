import sympy as sp

x = sp.symbols('x')
L1 = (x - 2)*(x - 3) / 2
L2 = -(x - 1)*(x - 3)
L3 = (x - 1)*(x - 2) / 2

P = 1 * L1 + 4 * L2 + 9 * L3
P_simplified = sp.expand(P)

print(P_simplified)
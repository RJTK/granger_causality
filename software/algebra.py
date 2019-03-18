import numpy as np
import matplotlib.pyplot as plt

from sympy import (Matrix, Symbol, MatrixSymbol, pprint, eye, tensorproduct,
                   linsolve, simplify, summation, oo, symbols, lambdify)

# In the system x(t) = Bx(t - 1) + v(t) where B is lower triangular,
# is it possible that the pairwise error from x_2 to x_1 is lower than from x_1 to x_2
# even though there is no direct connection?  What about the GCG likelihood ratio?

def mar18():
    a, b, c = symbols("a, b, c")
    B = Matrix([[0, 0, c], [a, 0, 0], [0, b, 0]])

    B = B.subs({b: 0.5, a: 0.5})
    eig = list(simplify(abs(e)) for e in B.eigenvals().keys())

    lmbda = lambdify(c, eig[0])
    num_a = np.linspace(0, 10, 10000)
    L = lmbda(num_a)

    plt.plot(num_a, L)
    plt.hlines(1, num_a[0], num_a[-1])
    plt.show()

    a, b, c, d = symbols("a, b, c, d")
    B = Matrix([[0, 0, 0, d],
                [a, 0, 0, 0],
                [0, b, 0, d],
                [0, c, 0, 0]])
    return

def feb24():
    # No idea about the actual date.
    xtm1 = MatrixSymbol("x(t - 1)", n=2, m=1)
    vt = MatrixSymbol("v(t)", n=2, m=1)

    b1 = Symbol("b_1")
    b2 = Symbol("b_2")
    a = Symbol("a")
    B = Matrix([[b1, 0], [a, b2]])

    sv1 = Symbol("sigma_v(1)^2", positive=True)
    sv2 = Symbol("sigma_v(2)^2", positive=True)
    Sv = Matrix([[sv1, 0], [0, sv2]])

    Rx0 = MatrixSymbol("R_x(0)", n=2, m=2)
    eq = Rx0 - (Sv + B * Rx0 * B.T)

    res = linsolve((eq[0, 0], eq[0, 1], eq[1, 0], eq[1, 1]),
                   Rx0[0, 0], Rx0[0, 1], Rx0[1, 0], Rx0[1, 1])

    Rx00, Rx01, Rx10, Rx11 = [simplify(v) for v in res.args[0]]
    Rx0 = Matrix([[Rx00, Rx01], [Rx10, Rx11]])
    Rx1 = simplify(B * Rx0)

    # Is this right?
    e00 = simplify(Rx0[0, 0] - (Rx1[0, 1]**2) / Rx0[0, 0])
    e = simplify(Rx0 - Rx1 * (Rx0.inv()) * Rx1.T)


    # System with 3 variables
    rho = Symbol("rho")
    a = Symbol("a")

    B = Matrix([[rho, 0, 0], [a, rho, 0], [a, 0, rho]])

    Rx0 = MatrixSymbol("R_x(0)", n=3, m=3)
    sv1 = Symbol("sigma_v(1)^2", positive=True)
    sv2 = Symbol("sigma_v(2)^2", positive=True)
    sv3 = Symbol("sigma_v(3)^2", positive=True)
    Sv = Matrix([[sv1, 0, 0], [0, sv2, 0], [0, 0, sv3]])

    k = Symbol("k")
    Bpk = B**k
    Z = Bpk * Sv * Bpk.T
    Rx0 = simplify(summation(Z, (k, 0, oo)))

    # Extract piecewise portion for |rho| < 1
    Rx0 = simplify(
        Matrix([[Rx0[i, j].args[0][0] for j in range(3)] for i in range(3)]))

    Rx1 = simplify(B * Rx0)
    Rx2 = simplify(B * Rx1)

    sv = Symbol("sigma_v^2")
    Rx0 = Rx0.subs({sv1: sv, sv2: sv, sv3: sv})
    Rx1 = Rx1.subs({sv1: sv, sv2: sv, sv3: sv})
    Rx2 = Rx2.subs({sv1: sv, sv2: sv, sv3: sv})

    i, j = 1, 2

    Rx0_sub = simplify(Rx0[[i, j], [i, j]])
    Rx1_sub = simplify(Rx1[[i, j], [i, j]])

    E_sub = simplify(Rx0_sub - Rx1_sub * (Rx0_sub.inv()) * Rx1_sub.T)
    E11 = simplify(Rx0[i, i] - Rx1[i, i]**2 / Rx0[i, i])
    E22 = simplify(Rx0[j, j] - Rx1[j, j]**2 / Rx0[j, j])

    pprint(E_sub.subs({rho: 0.5, a: 1.0, sv: 1.0}))
    pprint(E11.subs({rho: 0.5, a: 1.0, sv: 1.0}))
    pprint(E22.subs({rho: 0.5, a: 1.0, sv: 1.0}))
    return

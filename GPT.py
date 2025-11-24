import numpy as np
from scipy.optimize import linprog
import pandas as pd

def solve_flux(prod, demand, qmax):
    prod = np.array(prod)
    demand = np.array(demand)
    N = len(prod)

    n_q = N*(N-1)
    offset_r_pos = n_q
    offset_r_neg = n_q + N
    nvar = n_q + 2*N

    c = np.zeros(nvar)
    c[offset_r_neg: offset_r_neg + N] = 1.0

    A_eq = []
    b_eq = []

    def q_index(i, j):
        if i == j:
            raise ValueError("i should not equal j")
        return i*(N-1) + (j if j < i else j - 1)

    for i in range(N):
        row = np.zeros(nvar)
        for j in range(N):
            if i != j:
                row[q_index(i, j)] -= 1   # outflows
        for j in range(N):
            if i != j:
                row[q_index(j, i)] += 1   # inflows

        # r = r_pos - r_neg
        row[offset_r_pos + i] = 1
        row[offset_r_neg + i] = -1

        A_eq.append(row)
        b_eq.append(prod[i] - demand[i])

    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)

    Q = prod - demand

    # --- BOUNDS ---
    bounds = []

    # 1) bounds for q variables in SAME order as q_index enumeration (i then j)
    for i in range(N):
        for j in range(N):
            if i != j:
                bounds.append((0, qmax))

    # 2) bounds for ALL r_pos (i = 0..N-1)
    for i in range(N):
        if Q[i] >= 0:
            bounds.append((0, None))   # r_pos free >=0
        else:
            bounds.append((0, 0))      # r_pos forced to 0

    # 3) bounds for ALL r_neg (i = 0..N-1)
    for i in range(N):
        if Q[i] >= 0:
            bounds.append((0, 0))      # r_neg forced to 0
        else:
            bounds.append((0, None))   # r_neg free >=0

    # Solve LP
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if not res.success:
        raise RuntimeError("Optimization failed:", res.message)

    x = res.x

    # Reconstruct q using q_index (robuste)
    q = np.zeros((N, N))
    idx = 0
    for j in range(N):
        for i in range(N):
            if i != j:
                q[i][j] = x[idx]
                idx += 1

    r_pos = x[offset_r_pos:offset_r_pos + N]
    r_neg = x[offset_r_neg:offset_r_neg + N]

    # (optionnel) nettoyer tout petit négatif dû à la numérique
    eps = 1e-9
    r_pos = np.where(np.abs(r_pos) < eps, 0.0, r_pos)
    r_neg = np.where(np.abs(r_neg) < eps, 0.0, r_neg)

    return q, r_pos, r_neg


# test à un instant t
prod  = [10,  5,  7, 9]
demand = [6, 10, 7, 4]
qmax = 4

q, r_pos, r_neg = solve_flux(prod, demand, qmax)

print("Flux :\n", q)
print("r⁺ :", r_pos)
print("r⁻ :", r_neg)


#production du pays A pour tous les instants de 0 à N//3 - 1
#prodA = pd.read_csv('prodA.csv')

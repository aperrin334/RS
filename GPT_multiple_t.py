import numpy as np
from scipy.optimize import linprog

# prod : vecteur avec l'entièreté des prod à chaque temps t pour chaque pays (taille = N*T)
# demand : vecteur avec l'entièreté des demand à chaque temps t pour chaque pays (taille = N*T)
# qmax : capacité maximale de flux entre deux pays
# N : nombre de pays
# T : nombre de périodes

def solve_flux(prod, demand, qmax, N):
    prod = np.array(prod)
    demand = np.array(demand)
    T = len(prod)/N

    n_q = N*(N-1)*T
    offset_r_pos = n_q
    offset_r_neg = n_q + N*T
    nvar = n_q + 2*N*T

    c = np.zeros(nvar)
    c[offset_r_neg: offset_r_neg + N*T] = 1.0

    A_eq = []
    b_eq = []

    def q_index(i, j, t):
        if i == j:
            raise ValueError("i should not equal j")
        return t*N*(N-1) + i*(N-1) + (j if j < i else j - 1)
    # construction matrice des contraintes
    # ici pour les qi,j,t
    for t in range(int(T)):
        for i in range(N):
            row = np.zeros(nvar)
            for j in range(N):
                if i != j:
                    row[q_index(i, j) + t*N*(N-1)] -= 1   # outflows
                    row[q_index(j, i) + t*N*(N-1)] += 1   # inflows

            # ici pour les r_pos,t - r_neg,t = r_t
            row[offset_r_pos + i + t*N] = 1
            row[offset_r_neg + i + t*N] = -1

            A_eq.append(row)
            b_eq.append(prod[i + t*N] - demand[i + t*N])

    A_eq = np.array(A_eq)
    print("A_eq:")
    print(A_eq)
    b_eq = np.array(b_eq)
    print("b_eq:")
    print(b_eq)

    Q = prod - demand

    # --- BOUNDS ---
    bounds = []

    # 1) bounds for q variables in SAME order as q_index enumeration (i then j)
    for t in range(int(T)):
        for i in range(N):
            for j in range(N):
                if i != j:
                    bounds.append((0, qmax))
    # 2) bounds for ALL r_pos (i = 0..N-1)
    for t in range(int(T)):
        for i in range(N):
            if Q[i + t*N] >= 0:
                bounds.append((0, None))   # r_pos free >=0
            else:
                bounds.append((0, 0))      # r_pos forced to 0

    # 3) bounds for ALL r_neg (i = 0..N-1)
    for t in range(int(T)):
        for i in range(N):
            if Q[i + t*N] >= 0:
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

    # (optionnel) nettoyer tout petit négatif dû à la numérqiue
    eps = 1e-9
    r_pos = np.where(np.abs(r_pos) < eps, 0.0, r_pos)
    r_neg = np.where(np.abs(r_neg) < eps, 0.0, r_neg)

    return q, r_pos, r_neg





# prod  = [10,  5,  7, 9]
# demand = [6, 10, 7, 4]
# qmax = 4

# q, r_pos, r_neg = solve_flux(prod, demand, qmax)

# print("Flux :\n", q)
# print("r⁺ :", r_pos)
# print("r⁻ :", r_neg)

prod2  = [10,  5,  7]
demand2 = [6, 10, 7]
qmax2 = 4

q, r_pos, r_neg = solve_flux(prod2, demand2, qmax2)

print("Flux :\n", q)
print("r⁺ :", r_pos)
print("r⁻ :", r_neg)

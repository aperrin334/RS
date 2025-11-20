import pulp
import numpy as np

def to_array_from_prod_dict(prod_dict):
    """Convertit le dict {t: value} en numpy array ordonné."""
    N = len(prod_dict)
    return np.array([prod_dict[t] for t in range(N)])

def energy_flows_linear(prodA, prodB, prodC, demandA, demandB, demandC, qmax):
    """
    Résout un modèle linéaire :
      prodX[t] - demandX[t] - exports + imports + defX[t] >= 0
    Variables:
      qXY[t] >= 0 flux de X vers Y
      defX[t] >= 0 déficit sur X (ce qu'on veut minimiser)
    Retour: dictionnaire avec statuts et vecteurs de flux/déficits.
    """
    # Assurer que les entrées sont numpy arrays et de même longueur
    prodA = np.asarray(prodA)
    prodB = np.asarray(prodB)
    prodC = np.asarray(prodC)
    demandA = np.asarray(demandA)
    demandB = np.asarray(demandB)
    demandC = np.asarray(demandC)
    assert len(prodA) == len(demandA)
    assert len(prodB) == len(demandB)
    assert len(prodC) == len(demandC)
    N = len(prodA)

    # Problème
    prob = pulp.LpProblem("Energy_flows_linear", pulp.LpMinimize)

    # Variables de flux (continues, >=0)
    qAB = pulp.LpVariable.dicts("qAB", range(N), lowBound=0)
    qAC = pulp.LpVariable.dicts("qAC", range(N), lowBound=0)
    qBA = pulp.LpVariable.dicts("qBA", range(N), lowBound=0)
    qBC = pulp.LpVariable.dicts("qBC", range(N), lowBound=0)
    qCA = pulp.LpVariable.dicts("qCA", range(N), lowBound=0)
    qCB = pulp.LpVariable.dicts("qCB", range(N), lowBound=0)

    # Variables de déficit (>=0) — ce que l'on minimise
    defA = pulp.LpVariable.dicts("defA", range(N), lowBound=0)
    defB = pulp.LpVariable.dicts("defB", range(N), lowBound=0)
    defC = pulp.LpVariable.dicts("defC", range(N), lowBound=0)

    # Objectif : minimiser la somme des déficits horaires (option : pondérer si besoin)
    prob += pulp.lpSum([defA[t] + defB[t] + defC[t] for t in range(N)])

    # Contraintes horaires (bilan) :
    # prodA - demandA - qAB - qAC + qBA + qCA + defA >= 0
    # -> defA >= demandA - (prodA - qAB - qAC + qBA + qCA)
    for t in range(N):
        prob += (prodA[t] - demandA[t]
                 - qAB[t] - qAC[t]
                 + qBA[t] + qCA[t]
                 + defA[t] >= 0), f"balance_A_{t}"
        prob += (prodB[t] - demandB[t]
                 - qBA[t] - qBC[t]
                 + qAB[t] + qCB[t]
                 + defB[t] >= 0), f"balance_B_{t}"
        prob += (prodC[t] - demandC[t]
                 - qCA[t] - qCB[t]
                 + qAC[t] + qBC[t]
                 + defC[t] >= 0), f"balance_C_{t}"

    # Contraintes de capacité des interconnexions (par heure)
    for t in range(N):
        prob += qAB[t] <= qmax, f"cap_qAB_{t}"
        prob += qAC[t] <= qmax, f"cap_qAC_{t}"
        prob += qBA[t] <= qmax, f"cap_qBA_{t}"
        prob += qBC[t] <= qmax, f"cap_qBC_{t}"
        prob += qCA[t] <= qmax, f"cap_qCA_{t}"
        prob += qCB[t] <= qmax, f"cap_qCB_{t}"

    # Résoudre
    prob.solve(pulp.PULP_CBC_CMD(msg=False))  # msg=True pour logs

    # Récupérer résultats (convertir en numpy arrays)
    def get_vec(var_dict):
        return np.array([var_dict[t].value() for t in range(N)], dtype=float)

    result = {
        'status': pulp.LpStatus[prob.status],
        'qAB': get_vec(qAB),
        'qAC': get_vec(qAC),
        'qBA': get_vec(qBA),
        'qBC': get_vec(qBC),
        'qCA': get_vec(qCA),
        'qCB': get_vec(qCB),
        'defA': get_vec(defA),
        'defB': get_vec(defB),
        'defC': get_vec(defC),
        'total_deficit': sum(get_vec(defA)) + sum(get_vec(defB)) + sum(get_vec(defC)),
    }
    return result

# ---------------------------
# Exemple d'utilisation avec tes variables originales
# ---------------------------
# Si ton `prod` était un dict {t: value}, convertir :
# prod_arr = to_array_from_prod_dict(prod)
# Puis découper en 3 régions (A, B, C) de longueur T comme tu voulais :
# prodA = prod_arr[0:T]; prodB = prod_arr[T:2*T]; prodC = prod_arr[2*T:3*T]
#
# Appel exemple (en supposant prod_arr et df_demand disponibles) :
#
# result = energy_flows_linear(
#     prodA = prod_arr[0:T],
#     prodB = prod_arr[T:2*T],
#     prodC = prod_arr[2*T:3*T],
#     demandA = df_demand['demande'].values[0:T],
#     demandB = df_demand['demande'].values[T:2*T],
#     demandC = df_demand['demande'].values[2*T:3*T],
#     qmax = 17e12
# )
#
# Puis afficher un résumé :
# print("Status:", result['status'])
# print("Total deficit (sum over all hours and zones):", result['total_deficit'])
# print("Sum flows qAB (GWh):", result['qAB'].sum())
# print("Quelques valeurs qAB[:10]:", result['qAB'][:10])
# print("defA[:10]:", result['defA'][:10])

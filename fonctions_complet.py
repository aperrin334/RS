import numpy as np
#from scipy.optimize import linprog
#import pandas as pd
import pulp
#import networkx as nx
#import matplotlib.pyplot as plt
#from ipywidgets import interact, IntSlider
#from joblib import Parallel, delayed

#=====================================================
#-----------------COMPLET-----------------------------
#=====================================================

# Définition du problème -----------------------------

def extract_results(prob, T, N,
                    phs_in, phs_out, phs_level,
                    ptg_in, ptg_out, ptg_level,
                    deficit, surplus, q, prod):

    phs_in_array    = np.zeros((T, N))
    phs_out_array   = np.zeros((T, N))
    phs_level_array = np.zeros((T, N))

    ptg_in_array    = np.zeros((T, N))
    ptg_out_array   = np.zeros((T, N))
    ptg_level_array = np.zeros((T, N))

    deficit_array = np.zeros((T, N))
    surplus_array = np.zeros((T,N))
    
    q_array = np.zeros((T, N, N))
    
    for t in range(T):
        for i in range(N):
            phs_in_array[t, i]    = pulp.value(phs_in[t][i])
            phs_out_array[t, i]   = pulp.value(phs_out[t][i])
            phs_level_array[t, i] = pulp.value(phs_level[t][i])

            ptg_in_array[t, i]    = pulp.value(ptg_in[t][i])
            ptg_out_array[t, i]   = pulp.value(ptg_out[t][i])
            ptg_level_array[t, i] = pulp.value(ptg_level[t][i])

            deficit_array[t, i] = pulp.value(deficit[t][i])
            surplus_array[t, i] = pulp.value(surplus[t][i])

            for j in range(N):
                if i != j:
                    q_array[t, i, j] = pulp.value(q[t][i][j])

    return {        
        
        'status': pulp.LpStatus[prob.status],
        
        'prod' : prod,
        
        'phs_in': phs_in_array,
        'phs_out': phs_out_array,
        'phs_level': phs_level_array,

        'ptg_in': ptg_in_array,
        'ptg_out': ptg_out_array,
        'ptg_level': ptg_level_array,

        'deficit_final': deficit_array, #nom de variable 'deficit' modifié
        'surplus_final': surplus_array, #nom de variable 'surplus' modifié

        'echanges': q_array
    }


def optimize(
    wind_profile, solar_profile, demand,
    wind_cap, solar_cap, qmax_matrix,
    phs_capacity=180, phs_power=9.3, phs_eff=0.75,
    ptg_capacity=125000, ptg_init_rate=0.75, ptg_power_in=7.66, ptg_power_out=32.93,
    ptg_eff=0.4, penalisation=1e10, flux_eff=0.9
):
    
    """
    Optimise la stratégie de stockage sur l'année avec contrainte cyclique pour n pays
    
    Parameters:
    -----------
    solar_profile, wind_profile: séries temporelles des facteurs de charge
    demand: série temporelle de la demande
    wind_cap, solar_cap: capacités installées en GW
    phs_capacity: capacité de stockage PHS en GWh
    phs_power: puissance max PHS en GW
    phs_efficiency: rendement PHS (aller-retour)
    ptg_capacity: capacité de stockage P2G en GWh
    ptg_init_rate: taux initial de remplissage des stocks (entre 0 et 1)
    ptg_power_in: puissance max électrolyse en GW  
    ptg_power_out: puissance max reconversion en GW
    ptg_efficiency: rendement P2G (aller-retour)
    qmax_matrix : np.array (N, N), symétrique, diag = 0
    flux_eff : flux moins les pertes lors du transport

    """

    T, N = demand.shape

    # =====================
    # Production
    # =====================
    prod = np.zeros((T, N))
    for t in range(T):
        prod[t, :] = (
            wind_cap * wind_profile[t, :]
            + solar_cap * solar_profile[t, :]
        )

    # =====================
    # Problème
    # =====================
    prob = pulp.LpProblem("Energy_System", pulp.LpMinimize)

    # =====================
    # Variables
    # =====================

    phs_in    = pulp.LpVariable.dicts("phs_in",    (range(T), range(N)), lowBound=0)
    phs_out   = pulp.LpVariable.dicts("phs_out",   (range(T), range(N)), lowBound=0)
    phs_level = pulp.LpVariable.dicts("phs_level", (range(T), range(N)), lowBound=0)

    ptg_in    = pulp.LpVariable.dicts("ptg_in",    (range(T), range(N)), lowBound=0)
    ptg_out   = pulp.LpVariable.dicts("ptg_out",   (range(T), range(N)), lowBound=0)
    ptg_level = pulp.LpVariable.dicts("ptg_level", (range(T), range(N)), lowBound=0)

    deficit = pulp.LpVariable.dicts("deficit", (range(T), range(N)), lowBound=0)

    surplus = pulp.LpVariable.dicts("surplus", (range(T), range(N)), lowBound=0)

    q = pulp.LpVariable.dicts(
        "q", (range(T), range(N), range(N)), lowBound=0
    )

    # =====================
    # Fonction objectif
    # =====================

    prob += (
        pulp.lpSum(-ptg_level[T-1][i] for i in range(N))
        + penalisation * pulp.lpSum(deficit[t][i] for t in range(T) for i in range(N)) + 0.1*pulp.lpSum(q[t][i][j] for t in range(T) for i in range(N) for j in range(N) if i != j
    ))

    # =====================
    # Contraintes
    # =====================

    # Conditions initiales
    for i in range(N):
        prob += phs_level[0][i] == phs_capacity / 2
        prob += ptg_level[0][i] == ptg_init_rate * ptg_capacity

    for t in range(T):
        for i in range(N):

            inflow  = pulp.lpSum(q[t][j][i] for j in range(N) if j != i)
            outflow = pulp.lpSum(q[t][i][j] for j in range(N) if j != i)

            # Bilan énergie
            prob += (
                prod[t, i] 
                + phs_out[t][i] * phs_eff 
                + ptg_out[t][i] * ptg_eff 
                + inflow * flux_eff 
                + deficit[t][i]
                == 
                demand[t, i] 
                + phs_in[t][i] 
                + ptg_in[t][i] 
                + outflow 
                + surplus[t][i]
            )

            # Limites PHS
            prob += phs_in[t][i]  <= phs_power
            prob += phs_out[t][i] <= phs_power
            prob += phs_level[t][i] <= phs_capacity

            # Limites PtG
            prob += ptg_in[t][i]  <= ptg_power_in
            prob += ptg_out[t][i] <= ptg_power_out
            prob += ptg_level[t][i] <= ptg_capacity

            # Évolution des stocks
            if t > 0:
                prob += (
                    phs_level[t][i]
                    == phs_level[t-1][i]
                    + phs_in[t][i] * phs_eff
                    - phs_out[t][i]
                )

                prob += (
                    ptg_level[t][i]
                    == ptg_level[t-1][i]
                    + ptg_in[t][i] * ptg_eff
                    - ptg_out[t][i]
                )

            # Contraintes échanges
            for j in range(N):
                if i != j:
                    prob += q[t][i][j] <= qmax_matrix[i, j]
                else:
                    prob += q[t][i][j] == 0

    # =====================
    # Résolution
    # =====================
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    return extract_results(
    prob, T, N,
    phs_in, phs_out, phs_level,
    ptg_in, ptg_out, ptg_level,
    deficit, surplus, q, prod)

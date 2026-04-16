import numpy as np
#from scipy.optimize import linprog
import pandas as pd
import pulp
#import networkx as nx
#import matplotlib.pyplot as plt
#from ipywidgets import interact, IntSlider
from joblib import Parallel, delayed

#=====================================================
#-----------------SEPARE------------------------------
#=====================================================
# En fonction de V2 ou V3 les sorties ne sont pas exactement les mêmes donc il faut adapter l'affichage mais à terme on n'utilisera que la V3

# Optimisation stockage 1 pays -----------------------
def optimize_storage(prod, demand,
                    phs_capacity, phs_power, phs_eff,
                    ptg_capacity, ptg_init_rate, ptg_power_in, ptg_power_out, 
                    ptg_eff, penalisation):
    """
    Optimise la stratégie de stockage sur l'année avec contrainte cyclique
    
    Parameters:
    -----------
    solar_profile, wind_profile: séries temporelles des facteurs de charge
    demand: série temporelle de la demande
    wind_cap, solar_cap: capacités installées en GW
    phs_capacity: capacité de stockage PHS en GWh
    phs_power: puissance max PHS en GW
    phs_eff: rendement PHS (aller-retour)
    ptg_capacity: capacité de stockage P2G en GWh
    ptg_init_rate: taux initial de remplissage des stocks (entre 0 et 1)
    ptg_power_in: puissance max électrolyse en GW  
    ptg_power_out: puissance max reconversion en GW
    ptg_eff: rendement P2G (aller-retour)
    """
    
    T = len(demand)  # nombre d'heures
    
    # Création du problème
    prob = pulp.LpProblem("Storage_Strategy", pulp.LpMinimize)
    
    # Variables
    phs_in = pulp.LpVariable.dicts("phs_in", range(T), 0)  # Charge PHS
    phs_out = pulp.LpVariable.dicts("phs_out", range(T), 0)  # Décharge PHS
    phs_level = pulp.LpVariable.dicts("phs_level", range(T), 0)  # Niveau PHS
    
    ptg_in = pulp.LpVariable.dicts("ptg_in", range(T), 0)  # Charge P2G
    ptg_out = pulp.LpVariable.dicts("ptg_out", range(T), 0)  # Décharge P2G
    ptg_level = pulp.LpVariable.dicts("ptg_level", range(T), 0)  # Niveau P2G
    
    deficit = pulp.LpVariable.dicts("deficit", range(T), 0)  # Déficit
    surplus = pulp.LpVariable.dicts("surplus", range(T), lowBound=0) # Surplus
    
    
    # Fonction objectif : minimiser déficits
    prob += -ptg_level[T-1] + penalisation * pulp.lpSum(deficit[t] for t in range(T))
    
    # Contraintes
    
    # Niveaux initiaux
    prob += phs_level[0] == phs_capacity / 2  # PHS à 50%
    prob += ptg_level[0] == ptg_init_rate * ptg_capacity    # P2G plein à <ptg_init_rate> %
    
    # Contraintes cycliques : les niveaux finaux doivent être égaux aux niveaux initiaux
    #prob += phs_level[T-1] == phs_level[0]
    #prob += ptg_level[T-1] == ptg_level[0]
    
    for t in range(T):
        # Équilibre offre-demande
        prob += (
                prod[t] 
                + phs_out[t] * phs_eff 
                + ptg_out[t] * ptg_eff 
                + deficit[t]
                == 
                demand[t] 
                + phs_in[t] 
                + ptg_in[t]
                + surplus[t]
        )
        
        # Limites PHS
        prob += phs_in[t] <= phs_power
        prob += phs_out[t] <= phs_power
        prob += phs_level[t] <= phs_capacity
        
        # Limites P2G
        prob += ptg_in[t] <= ptg_power_in
        prob += ptg_out[t] <= ptg_power_out
        prob += ptg_level[t] <= ptg_capacity
        
        # Évolution des stocks
        if t > 0:
            prob += phs_level[t] == phs_level[t-1] + phs_in[t]*phs_eff - phs_out[t]
            prob += ptg_level[t] == ptg_level[t-1] + ptg_in[t]*ptg_eff - ptg_out[t]
       
        
    # Résolution
    prob.solve()
    
    return {
        'status': pulp.LpStatus[prob.status],
        'prod':prod,
        'phs_in': [phs_in[t].value() for t in range(T)],
        'phs_out': [phs_out[t].value() for t in range(T)],
        'phs_level': [phs_level[t].value() for t in range(T)],
        'ptg_in': [ptg_in[t].value() for t in range(T)],
        'ptg_out': [ptg_out[t].value() for t in range(T)],
        'ptg_level': [ptg_level[t].value() for t in range(T)],
        'deficit': [deficit[t].value() for t in range(T)],
        'surplus': [surplus[t].value() for t in range(T)]
    }

# Optimisation des échanges N pays -------------------
## Avec la V2 : -> J'ai enlevé l'assertion demandant des capa max symétriques entre les pays mais il faudra vérifier que le code le traite bien correctement
def solve_flux(surplus, deficit, qmax_matrix, flux_eff, penalisation):
    """
    Résout le flux optimal pour N pays et T périodes, avec PuLP.

    surplus : np.array (T, N)
    deficit : np.array (T, N)
    qmax_matrix : np.array (N, N), symétrique, diag = 0
    """

    T, N = surplus.shape

    # Sécurité : contrôle taille qmax_matrix
    assert qmax_matrix.shape == (N, N), "qmax_matrix doit être de taille (N, N)"
    #assert np.allclose(qmax_matrix, qmax_matrix.T), "qmax_matrix doit être symétrique"
    assert np.all(np.diag(qmax_matrix) == 0), "La diagonale de qmax_matrix doit être nulle"

    Q = surplus - deficit

    # Problème d'optimisation
    prob = pulp.LpProblem("Optimal_Flux", pulp.LpMinimize)

    # Flux inter-pays q[t,i,j]
    q = {}
    for t in range(T):
        for i in range(N):
            for j in range(N):
                if i != j:
                    q[t, i, j] = pulp.LpVariable(
                        f"q_{t}_{i}_{j}",
                        lowBound=0,
                        upBound=float(qmax_matrix[i, j])
                    )

    # Variables r_pos et r_neg
    r_pos, r_neg = {}, {}
    for t in range(T):
        for i in range(N):
            if Q[t, i] >= 0:
                r_pos[t, i] = pulp.LpVariable(f"r_pos_{t}_{i}", lowBound=0)
                r_neg[t, i] = pulp.LpVariable(f"r_neg_{t}_{i}", lowBound=0, upBound=0)
            else:
                r_pos[t, i] = pulp.LpVariable(f"r_pos_{t}_{i}", lowBound=0, upBound=0)
                r_neg[t, i] = pulp.LpVariable(f"r_neg_{t}_{i}", lowBound=0)

    # Fonction objectif : minimiser les déficites et maximiser les restes positifs
    prob += pulp.lpSum(penalisation*r_neg[t, i]-r_pos[t, i] for t in range(T) for i in range(N))

    # Contraintes de bilan
    for t in range(T):
        for i in range(N):
            inflow = pulp.lpSum(q[t, j, i] for j in range(N) if j != i)
            outflow = pulp.lpSum(q[t, i, j] for j in range(N) if j != i)
            prob += (
                r_pos[t, i] - r_neg[t, i]
                == surplus[t, i] - deficit[t, i] + flux_eff * inflow - outflow
            )

    # Résolution
    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    # Extraction solutions
    q_array = np.zeros((T, N, N))
    new_surplus = np.zeros((T, N))
    new_deficit= np.zeros((T, N))

    for t in range(T):
        for i in range(N):
            new_surplus[t, i] = pulp.value(r_pos[t, i])
            new_deficit[t, i] = pulp.value(r_neg[t, i])
            for j in range(N):
                if i != j:
                    q_array[t, i, j] = pulp.value(q[t, i, j])

    return {
        'echanges' : q_array,
        'surplus' : new_surplus,
        'deficit' : new_deficit
    }
## Avec la V3 :
def solve_flux_interval(dispo, qmax_matrix, flux_eff, targets, penalisation=1e10):
    T, N = dispo.shape
    prob = pulp.LpProblem("Flux_Interval_Target", pulp.LpMinimize)

    # 1. Variables de flux : On ne crée que si i != j
    q = {}
    for t in range(T):
        for i in range(N):
            for j in range(N):
                if i != j:
                    q[t, i, j] = pulp.LpVariable(
                        f"q_{t}_{i}_{j}",
                        lowBound=0,
                        upBound=float(qmax_matrix[i, j])
                    )

    # 2. Variables de dépassement (Slacks)
    # over : surplus qui dépasse target_max
    # under : déficit qui tombe sous target_min
    over = pulp.LpVariable.dicts("over", (range(T), range(N)), lowBound=0)
    under = pulp.LpVariable.dicts("under", (range(T), range(N)), lowBound=0)

    # 3. Fonction Objectif
    # Minimiser les dépassements d'intervalle. 
    # Optionnel : + 0.001 * q pour éviter les flux circulaires inutiles
    prob += pulp.lpSum((penalisation *under[t][i] + over[t][i]) for t in range(T) for i in range(N)) + pulp.lpSum((q[t, i, j]) for t in range(T) for i in range(N) for j in range(N) if i != j)

    # 4. Contraintes de bilan
    for t in range(T):
        for i in range(N):
            t_min, t_max = targets[i]
            
            # Somme des flux entrants et sortants (uniquement si la clé existe)
            inflow = pulp.lpSum(q[t, j, i] for j in range(N) if i != j)
            outflow = pulp.lpSum(q[t, i, j] for j in range(N) if i != j)
            
            # Balance finale = balance initiale + imports_nets
            balance_f = dispo[t, i] + (flux_eff * inflow) - outflow
            
            # Contraintes d'intervalle "souples" (Soft Constraints)
            # balance_f - over <= target_max  => si balance > max, over devient > 0
            # balance_f + under >= target_min => si balance < min, under devient > 0
            prob += balance_f - over[t][i] <= t_max
            prob += balance_f + under[t][i] >= t_min

    # Résolution
    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    # Extraction des résultats
    q_res = np.zeros((T, N, N))
    final_balances = np.zeros((T, N))
    
    for t in range(T):
        for i in range(N):
            for j in range(N):
                if i != j:
                    q_res[t, i, j] = pulp.value(q[t, i, j])
            
            # Recalcul de la balance finale réelle
            inf = sum(q_res[t, j, i] for j in range(N) if i != j)
            outf = sum(q_res[t, i, j] for j in range(N) if i != j) # erreur ici corrigée dessous
            # Correction :
            outf = sum(q_res[t, i, j] for j in range(N) if i != j)
            final_balances[t, i] = dispo[t, i] + (flux_eff * inf) - outf

    return {
        'flux': q_res,
        'balances_finales': final_balances
    }

# Combiner les deux optimisations --------------------
## Avec la V2 :
def optimize2(
    wind_profile, solar_profile, demand,
    wind_cap, solar_cap, qmax_matrix,
    phs_capacity=180, phs_power=9.3, phs_eff=0.75,
    ptg_capacity=125000, ptg_init_rate=0.75, ptg_power_in=7.66, ptg_power_out=32.93,
    ptg_eff=0.4, penalisation=1e10, flux_eff=0.9
):

    T, N = demand.shape
    
    # 1 - Calcul de la production
    
    prod = (
        wind_cap * wind_profile
        + solar_cap * solar_profile
    )
    
    # 2 - Calculs des échanges 

    result1=solve_flux(prod, demand, qmax_matrix, flux_eff, penalisation)
    
    echanges=result1['echanges']
    surplus_avant=result1['surplus']
    deficit_avant=result1['deficit']
    
    # 3 - Optimisation du stockage pour chaque pays en parralèle
    
    def run_one_country(p):
        return optimize_storage(
            surplus_avant[:, p], deficit_avant[:, p], phs_capacity, phs_power, phs_eff, 
            ptg_capacity, ptg_init_rate, ptg_power_in, ptg_power_out, ptg_eff, penalisation
        )

    result2 = Parallel(n_jobs=-1)(delayed(run_one_country)(p) for p in range(N))

    phs_in, phs_out, phs_level = np.zeros((T, N)), np.zeros((T, N)), np.zeros((T, N))
    ptg_in, ptg_out, ptg_level = np.zeros((T, N)), np.zeros((T, N)), np.zeros((T, N))
    deficit, surplus = np.zeros((T, N)), np.zeros((T, N))

    # 4 - Rassemblement des résultats
    for p, res in enumerate(result2):
        phs_in[:, p] = res['phs_in']
        phs_out[:, p] = res['phs_out']
        phs_level[:, p] = res['phs_level']
        ptg_in[:, p] = res['ptg_in']
        ptg_out[:, p] = res['ptg_out']
        ptg_level[:, p] = res['ptg_level']
        deficit[:, p] = res['deficit']
        surplus[:, p] = res['surplus']


    return {
        
        'prod' : prod,
        
        'phs_in': phs_in,
        'phs_out': phs_out,
        'phs_level': phs_level,

        'ptg_in': ptg_in,
        'ptg_out': ptg_out,
        'ptg_level': ptg_level,
        
        'deficit avant echange' : deficit_avant,
        'surplus avant echange' : surplus_avant,

        'deficit': deficit,
        'surplus': surplus,

        'Echanges': echanges
    }
## Avec la V3 :
def optimize3(
    wind_profile, solar_profile, demand,
    wind_cap, solar_cap, qmax_matrix,
    phs_capacity=180, phs_power=9.3, phs_eff=0.75,
    ptg_capacity=125000, ptg_init_rate=0.75, ptg_power_in=7.66, ptg_power_out=32.93,
    ptg_eff=0.4, penalisation=1e10, flux_eff=0.9
):

    T, N = demand.shape
    
    # 1 - Calcul de la production et des disponibilitées
    prod = (wind_cap * wind_profile + solar_cap * solar_profile)
    dispo_initiale = prod - demand

    # 2 - Définition de l'intervalle cible pour chaque pays
    target_min = -(phs_power + ptg_power_in)  # Capacité à absorber le surplus
    target_max = (phs_power + ptg_power_out)  # Capacité à combler le déficit
    
    # On crée la liste des targets pour solve_flux_interval
    targets = [(target_min, target_max) for _ in range(N)]

    # 3 - Calcul des échanges pour rentrer dans ces intervalles
    result_flux = solve_flux_interval(
        dispo_initiale, 
        qmax_matrix, 
        flux_eff, 
        targets, 
        penalisation
    )

    echanges = result_flux['flux']
    balance_apres_echanges = result_flux['balances_finales']

    # 4 - Optimisation du stockage pour chaque pays
    # On sépare la balance en surplus/déficit pour rester compatible avec optimize_storage
    surplus_pour_stockage = np.maximum(0, balance_apres_echanges)
    deficit_pour_stockage = np.maximum(0, -balance_apres_echanges)

    def run_one_country(p):
        return optimize_storage(
            surplus_pour_stockage[:, p], 
            deficit_pour_stockage[:, p],
            phs_capacity, phs_power, phs_eff,
            ptg_capacity, ptg_init_rate, ptg_power_in, ptg_power_out, 
            ptg_eff, penalisation
        )
        


    result2 = Parallel(n_jobs=-1)(delayed(run_one_country)(p) for p in range(N))

    # 5 - Rassemblement des résultats
    phs_in, phs_out, phs_level = np.zeros((T, N)), np.zeros((T, N)), np.zeros((T, N))
    ptg_in, ptg_out, ptg_level = np.zeros((T, N)), np.zeros((T, N)), np.zeros((T, N))
    deficit_final, surplus_final = np.zeros((T, N)), np.zeros((T, N))

    for p, res in enumerate(result2):
        phs_in[:, p] = res['phs_in']
        phs_out[:, p] = res['phs_out']
        phs_level[:, p] = res['phs_level']
        ptg_in[:, p] = res['ptg_in']
        ptg_out[:, p] = res['ptg_out']
        ptg_level[:, p] = res['ptg_level']
        deficit_final[:, p] = res['deficit']
        surplus_final[:, p] = res['surplus']

    return {
        'prod': prod,

        'phs_in': phs_in,
        'phs_out': phs_out,
        'phs_level': phs_level,

        'ptg_in': ptg_in,
        'ptg_out': ptg_out,
        'ptg_level': ptg_level,

        'balance_initiale': dispo_initiale,
        'balance_apres_echanges': balance_apres_echanges,

        'deficit_final': deficit_final,
        'surplus_final': surplus_final,

        'echanges': echanges
    }
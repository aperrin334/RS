from unittest import result

import numpy as np
import matplotlib.pyplot as plt

'''
Format du résultat de COMPLET :
    'status', 'prod', 
    'phs_in', 'phs_out', 'phs_level', 
    'ptg_in', 'ptg_out', 'ptg_level', 
    'deficit', 'surplus', 'echanges'
    
Format du résultat de SEPARE V3 :
    'prod', 
    'phs_in', 'phs_out', 'phs_level', 
    'ptg_in', 'ptg_out', 'ptg_level',
    'balance_initiale', 'balance_apres_echanges', 
    'deficit_final', 'surplus_final', 'echanges'
'''

def plot_graphs(result, semaine=5) :
    '''
    result : dictionnaire solution d'un modèle (complet ou separe)
    T_plot : nombre d'heures représentées sur les graphes
    semaine : semaine représentée sur les graphes "zoomés"
    '''

    # Sécurité pour gérer les deux noms de clés possibles
    if 'deficit' in result and 'deficit_final' not in result:
        result['deficit_final'] = result['deficit']
    if 'surplus' in result and 'surplus_final' not in result:
        result['surplus_final'] = result['surplus']

    T_plot=result['phs_level'].shape[0]
    N = result['phs_level'].shape[1]

    # --- Flux ---
    q = result["echanges"]  # (T, N, N)
    sent = np.sum(q, axis=2)
    received = np.sum(q, axis=1)

    # =====================================================
    # GRAPHE GLOBAL (<T_plot> heures)
    # =====================================================

    fig, axes = plt.subplots(6, N, figsize=(22, 14), sharex=True)

    for i in range(N):
        # PHS Level
        axes[0, i].plot(result['phs_level'][:T_plot, i], color='blue')
        if i == 0: axes[0, i].set_ylabel("Stock PHS (MWh)")
        axes[0, i].set_title(f"Pays {i+1}")

        # PtG Level
        axes[1, i].plot(result['ptg_level'][:T_plot, i], color='green')
        if i == 0: axes[1, i].set_ylabel("Stock Gaz (MWh)")
        
        # Déficit (énergie non servie)
        axes[2, i].plot(result['deficit_final'][:T_plot, i], color='red') # 'deficit' pour la v3 sur 1 an ?
        if i == 0: axes[2, i].set_ylabel("Déficit (MW)")

        # Flux sortants
        axes[3, i].plot(sent[:T_plot, i], color='orange')
        if i == 0: axes[3, i].set_ylabel("Export (MW)")

        # Flux entrants
        axes[4, i].plot(received[:T_plot, i], color='purple')
        if i == 0: axes[4, i].set_ylabel("Import (MW)")

        # Surplus (perdu ou non utilisé)
        axes[5, i].plot(result['surplus_final'][:T_plot, i], color='gray') # 'surplus' dans la v3 1 an
        if i == 0: axes[5, i].set_ylabel("Surplus (MW)")
        axes[5, i].set_xlabel("Temps (Heures)")

    plt.suptitle(f"Résultats sur {T_plot} heures", fontsize=16)
    plt.tight_layout()
    plt.show()


    # =====================================================
    # ZOOM SEMAINE
    # =====================================================
    
    T0 = 168 * (semaine - 1)
    T1 = T0 + 168

    fig, axes = plt.subplots(6, N, figsize=(22, 14), sharex=True)

    for i in range(N):

        axes[0, i].plot(result['phs_level'][T0:T1, i], color='blue')
        if i == 0: axes[0, i].set_ylabel("PHS")
        axes[0, i].set_title(f"Pays {i+1}")

        axes[1, i].plot(result['ptg_level'][T0:T1, i], color='green')
        if i == 0: axes[1, i].set_ylabel("PtG")

        axes[2, i].plot(result['deficit_final'][T0:T1, i], color='red')
        if i == 0: axes[2, i].set_ylabel("Déficit final")

        axes[3, i].plot(sent[T0:T1, i], color='orange')
        if i == 0: axes[3, i].set_ylabel("Envoyé")

        axes[4, i].plot(received[T0:T1, i], color='purple')
        if i == 0: axes[4, i].set_ylabel("Reçu")
        
        axes[5, i].plot(result['surplus_final'][T0:T1, i], color='gray')
        if i == 0: axes[5, i].set_ylabel("Surplus final")
        axes[5, i].set_xlabel("Heures semaine")

    plt.suptitle(f"Zoom — Semaine {semaine}", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_global_graph(result, demand_N, wind_N, wind_cap_N, solar_N, solar_cap_N, ptg_eff, phs_eff, semaine=5) :
    # ===============================
    # Flux échanges
    # ===============================

    # Sécurité pour gérer les deux noms de clés possibles
    if 'deficit' in result and 'deficit_final' not in result:
        result['deficit_final'] = result['deficit']
    if 'surplus' in result and 'surplus_final' not in result:
        result['surplus_final'] = result['surplus']
    
    T0 = 168 * (semaine - 1)
    T1 = T0 + 168
    time_axis = np.arange(T0, T1)

    q = result["echanges"]
    N = result['phs_level'].shape[1]

    imports = np.sum(q, axis=1)
    exports = np.sum(q, axis=2)

    fig, axes = plt.subplots(N, 1, figsize=(22, 16), sharex=True)

    for i in range(N):

        # --- Production ---
        wind_i = (wind_N*wind_cap_N)[T0:T1, i]
        solar_i = (solar_N*solar_cap_N)[T0:T1, i]

        # --- Déstockage ---
        phs_dis = phs_eff * result['phs_out'][T0:T1, i]
        ptg_dis = ptg_eff * result['ptg_out'][T0:T1, i]

        # --- Stockage (négatif) ---
        phs_ch = -result['phs_in'][T0:T1, i]
        ptg_ch = -result['ptg_in'][T0:T1, i]

        # --- Echanges ---
        import_i = imports[T0:T1, i]
        export_i = -exports[T0:T1, i]   # NEGATIF

        deficit_i = result['deficit_final'][T0:T1, i]
        demand_i = demand_N[T0:T1, i]

        # ===== STACK POSITIF =====
        stack_pos = [
            wind_i,
            solar_i,
            phs_dis,
            ptg_dis,
            import_i,
            deficit_i
        ]

        labels_pos = [
            'Wind',
            'PV',
            'PHS discharge',
            'PtG discharge',
            'Imports',
            'Deficit'
        ]

        axes[i].stackplot(time_axis, stack_pos, alpha=0.85)

        # ===== STACK NEGATIF =====
        stack_neg = [
            phs_ch,
            ptg_ch,
            export_i
        ]

        axes[i].stackplot(time_axis, stack_neg, alpha=0.85)

        # --- Demande ---
        axes[i].plot(time_axis,
                    demand_i,
                    color='black',
                    linewidth=2,
                    label='Demand')

        axes[i].axhline(0, linewidth=1)

        axes[i].set_title(f"Pays {i+1}")
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

        if i == 0:
            axes[i].legend(labels_pos + ['PHS charge','PtG charge','Exports','Demand'],
                        loc='upper left', fontsize=10)

    axes[-1].set_xlabel("Time (hours)")
    plt.suptitle(f"Bilan énergétique — Semaine {semaine}", fontsize=16)
    plt.tight_layout()
    plt.show()







def plot_energy_balance(
    result,
    demand_N,
    wind_N,
    wind_cap_N,
    solar_N,
    solar_cap_N,
    T0,
    T1,
    semaine,
    ptg_eff=0.4,
    phs_eff=0.75,
    show_surplus=False,
    show_max_storage=False,
    max_storage_phs=None,
    max_storage_ptg=None,
    figsize=(16, 16)
):
    """
    Affiche le bilan énergétique pour N pays sur une période donnée.

    Paramètres :
    - result : dictionnaire contenant les résultats du modèle (echanges, phs_in, phs_out, etc.)
    - demand_N, wind_N, solar_N, wind_cap_N, solar_cap_N : données d'entrée
    - T0, T1 : indices de temps pour la période à afficher
    - semaine : numéro de semaine pour le titre
    - ptg_eff, phs_eff : rendements de stockage
    - show_surplus : si True, affiche le surplus avant échange
    - show_max_storage : si True, affiche les capacités max de stockage
    - max_storage_phs, max_storage_ptg : valeurs des capacités max (si show_max_storage=True)
    - figsize : taille de la figure
    """

    # Sécurité pour gérer les deux noms de clés possibles
    if 'deficit' in result and 'deficit_final' not in result:
        result['deficit_final'] = result['deficit']
    if 'surplus' in result and 'surplus_final' not in result:
        result['surplus_final'] = result['surplus']

    N = demand_N.shape[1]
    time_axis = np.arange(T0, T1)
    q = result["echanges"]

    imports = np.sum(q, axis=1)
    exports = np.sum(q, axis=2)

    fig, axes = plt.subplots(N, 1, figsize=figsize, sharex=True)

    for i in range(N):

        # --- Production ---
        wind_i = (wind_N*wind_cap_N)[T0:T1, i]
        solar_i = (solar_N*solar_cap_N)[T0:T1, i]

        # --- Déstockage ---
        phs_dis = phs_eff * result['phs_out'][T0:T1, i]
        ptg_dis = ptg_eff * result['ptg_out'][T0:T1, i]

        # --- Stockage (négatif) ---
        phs_ch = -result['phs_in'][T0:T1, i]
        ptg_ch = -result['ptg_in'][T0:T1, i]

        # --- Echanges ---
        import_i = imports[T0:T1, i]
        export_i = -exports[T0:T1, i]

        deficit_i = result['deficit_final'][T0:T1, i]
        demand_i = demand_N[T0:T1, i]

        # --- Stack positif ---
        stack_pos = [
            wind_i,
            solar_i,
            phs_dis,
            ptg_dis,
            import_i,
            deficit_i
        ]

        labels_pos = [
            'Wind',
            'PV',
            'PHS discharge',
            'PtG discharge',
            'Imports',
            'Deficit'
        ]

        axes[i].stackplot(time_axis, stack_pos, alpha=0.85)

        # --- Stack négatif ---
        stack_neg = [
            phs_ch,
            ptg_ch,
            export_i
        ]

        axes[i].stackplot(time_axis, stack_neg, alpha=0.85)

        # --- Demande ---
        axes[i].plot(time_axis, demand_i, color='black', linewidth=2, label='Demand')

        # --- Surplus avant échange (optionnel) ---
        if show_surplus:
            surplus_i = -result["surplus avant echange"][T0:T1, i]
            axes[i].plot(time_axis, surplus_i, color='blue', linewidth=2, linestyle='--', label='Surplus avant échange')

        # --- Capacité max stockage (optionnel) ---
        if show_max_storage:
            axes[i].plot(time_axis, -max_storage_phs, color='red', linewidth=2, linestyle='--', label='Capa max PHS')
            axes[i].plot(time_axis, -max_storage_ptg, color='green', linewidth=2, linestyle='--', label='Capa max PtG')

        axes[i].axhline(0, linewidth=1)
        axes[i].set_title(f"Pays {i+1}")
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

        if i == 0:
            legend_labels = labels_pos + ['PHS charge','PtG charge','Exports','Demand']
            if show_surplus:
                legend_labels.append('Surplus')
            if show_max_storage:
                legend_labels.extend(['Capa max PHS', 'Capa max PtG'])
            axes[i].legend(legend_labels, loc='upper left', fontsize=10)

    axes[-1].set_xlabel("Time (hours)")
    plt.suptitle(f"Bilan énergétique — Semaine {semaine}", fontsize=16)
    plt.tight_layout()
    plt.show()
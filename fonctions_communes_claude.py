import numpy as np
#from scipy.optimize import linprog
import pandas as pd
#import pulp
#import networkx as nx
#import matplotlib.pyplot as plt
#from ipywidgets import interact, IntSlider
#from joblib import Parallel, delayed


#=====================================================
#-----------------DONNEES-----------------------------
#=====================================================

def charge_demand() :
    # Configuration des années et types
    years = [2019, 2020, 2021, 2022, 2023]
    N = len(years)

    df_demand = pd.read_csv('./demand2050_ADEME.csv', header=None)
    df_demand.columns = ["heures", "demande"]
    # Pour la demande, on s'assure aussi qu'elle fait 8760
    demand_values = df_demand['demande'].values[:8760]
    demand_N = np.tile(demand_values * 2.5, (N, 1))

    return demand_N

def charge_demand_multi() :

    demand = []
    annual_demand = []

    countries = pd.read_csv("./data_exchange/areas.csv", header=None)
    for country in countries.values:
        df_demand = pd.read_csv(f'./data_demand/demand_{country[0]}.csv', header=None)
        # Les CSV n'ont pas de header : col 0 = heures, col 1 = demande
        df_annual_demand = df_demand.iloc[:, 1].sum()
        df_demand_values = df_demand.iloc[:8760, 1].values * 2.5
        demand.append(df_demand_values)
        annual_demand.append(df_annual_demand)
    demand = np.array(demand)
    annual_demand = np.array(annual_demand)

    return demand, annual_demand

# Chargement des données pour 1 pays du code jouet et définition des capacités requises
def charge_data() :
    # Chargement des données
    df_solar = pd.read_csv('solar.csv')
    df_wind = pd.read_csv('wind.csv')
    df_demand = pd.read_csv('demand2050_ADEME.csv', header=None)
    df_demand.columns = ["heures", "demande"]

    # Calcul consommation annuelle
    annual_demand = df_demand['demande'].sum()

    # Calcul production potentielle pour 1GW installé
    annual_solar_per_gw = df_solar['facteur_charge'].sum()
    annual_wind_per_gw = df_wind['facteur_charge'].sum()

    # Pour avoir 100% de surproduction et un mix 50-50
    target_production = annual_demand * 3.5
    wind_capacity = target_production/(2 * annual_wind_per_gw)
    solar_capacity = target_production/(2 * annual_solar_per_gw)

    print(f"Capacités requises: éolien = {wind_capacity:.1f}GW, solaire = {solar_capacity:.1f}GW")

    return df_demand, df_solar, df_wind, solar_capacity, wind_capacity

def charge_data_5years(multi = True):
    # Configuration des années et types
    years = [2018, 2019, 2023, 2020, 2021, 2022, 2023]
    N = len(years)

    solar_profiles = []
    wind_profiles = []
    wind_caps = []
    solar_caps = []

    if multi :
        demand_N, annual_demand_N = charge_demand_multi()
        demand_N = demand_N[:N,]
        annual_demand_N = annual_demand_N[:N,]
    else :
        demand_N = charge_demand()
        annual_demand_N = demand_N.sum(axis=1)

    for year in years:
        # --- Traitement SOLAIRE ---
        df_s = pd.read_csv(f"./data_climix/{year}/solar_{year}.csv")
        s_vals = df_s['facteur_charge'].values
        
        # Force la taille à 8760
        solar_fc = np.zeros(8760)
        length_s = min(len(s_vals), 8760)
        solar_fc[:length_s] = s_vals[:length_s]
        solar_profiles.append(solar_fc)
        
        # --- Traitement ÉOLIEN ---
        df_w = pd.read_csv(f"./data_climix/{year}/wind_{year}.csv")
        w_vals = df_w['facteur_charge'].values
        
        # Initialisation du cumul éolien à 8760
        wind_fc = np.zeros(8760)
        length_w = min(len(w_vals), 8760)
        wind_fc[:length_w] = w_vals[:length_w]
        wind_profiles.append(wind_fc)
        
        # Calcul des capacités
        ann_solar_gw = solar_fc.sum()
        ann_wind_gw = wind_fc.sum()
        
        target_production = annual_demand_N[years.index(year)] * 3.5

        wind_caps.append(target_production / (2 * ann_wind_gw))
        solar_caps.append(target_production / (2 * ann_solar_gw))

    # Maintenant toutes les listes font exactement 8760, la conversion marchera :
    solar_N = np.array(solar_profiles).T
    wind_N = np.array(wind_profiles).T
    demand_N = demand_N.T

    wind_cap_N = np.array(wind_caps)
    solar_cap_N = np.array(solar_caps)


    print(f"\nDonnées prêtes : Matrice Solaire {solar_N.shape}, Matrice Éolienne {wind_N.shape}")

    return demand_N, solar_cap_N, wind_cap_N, solar_N, wind_N

def charge_data_multi():
    # Configuration des années et types
    countries = pd.read_csv("./data_exchange/areas.csv", header=None).squeeze("columns")
    N = len(countries)

    solar_profiles = []
    wind_profiles = []
    wind_caps = []
    solar_caps = []

    demand_N, annual_demand_N = charge_demand_multi()
    demand_N = demand_N[:N,]
    annual_demand_N = annual_demand_N[:N,]

    for i, country in enumerate(countries.values):
        # --- Traitement SOLAIRE ---
        # Les CSV n'ont pas de header : col 0 = index horaire, col 1 = facteur de charge
        df_s = pd.read_csv(f"./data_prod/countries_data/{country}/solar_{country}.csv", header=None)
        s_vals = df_s.iloc[:, 1].values
        
        # Force la taille à 8760
        solar_fc = np.zeros(8760)
        length_s = min(len(s_vals), 8760)
        solar_fc[:length_s] = s_vals[:length_s]
        solar_profiles.append(solar_fc)
        
        # --- Traitement ÉOLIEN ---
        df_w = pd.read_csv(f"./data_prod/countries_data/{country}/wind_{country}.csv", header=None)
        w_vals = df_w.iloc[:, 1].values
        
        # Initialisation du cumul éolien à 8760
        wind_fc = np.zeros(8760)
        length_w = min(len(w_vals), 8760)
        wind_fc[:length_w] = w_vals[:length_w]
        wind_profiles.append(wind_fc)
        
        # Calcul des capacités
        ann_solar_gw = solar_fc.sum()
        ann_wind_gw = wind_fc.sum()
        
        # Utilisation de l'index i (enumerate) plutôt que .index() pour éviter les doublons
        target_production = annual_demand_N[i] * 3.5

        if ann_wind_gw > 0:
            wind_caps.append(target_production / (2 * ann_wind_gw))
        else:
            wind_caps.append(0)

        if ann_solar_gw > 0:
            solar_caps.append(target_production / (2 * ann_solar_gw))
        else:
            solar_caps.append(0)

    # Maintenant toutes les listes font exactement 8760, la conversion marchera :
    solar_N = np.array(solar_profiles).T
    wind_N = np.array(wind_profiles).T
    demand_N = demand_N.T

    wind_cap_N = np.array(wind_caps)
    solar_cap_N = np.array(solar_caps)


    print(f"\nDonnées prêtes : Matrice Solaire {solar_N.shape}, Matrice Éolienne {wind_N.shape}")

    return demand_N, solar_cap_N, wind_cap_N, solar_N, wind_N

# divise le jeu de donnée en N pour simuler N pays (à terme on utilisera plutôt de vraies données)
def transfo_multipays(df_demand, df_solar, df_wind, solar_capacity, wind_capacity, N=5) :
    T = len(df_demand)
 
    assert T % N == 0, "T doit être divisible par N"

    demand_1 = df_demand['demande'].values*2.5 # shape (T,)
    demand_N = demand_1.reshape(N, T // N)

    solar_1 = df_solar['facteur_charge'].values
    solar_N = solar_1.reshape(N, T // N)

    wind_1  = df_wind['facteur_charge'].values
    wind_N  = wind_1.reshape(N, T // N)

    demand_N = demand_N.T    # shape (T/N, N)
    solar_N  = solar_N.T
    wind_N   = wind_N.T

    wind_cap_N  = np.full(N, wind_capacity)
    solar_cap_N = np.full(N, solar_capacity)

    return demand_N, solar_N, wind_N, solar_cap_N, wind_cap_N

# Fonction de création de la matrice de capacités max d'échange à partir des données
def capa_max_echanges() :
    pays =  pd.read_csv("./data_exchange/areas.csv", header=None)
    nb_pays_liste = len(pays.values)
    capmax = pd.read_csv("./data_exchange/links.csv",header=None, names=['a1','a2','links']).set_index(['a1','a2']).squeeze(axis=1)
    qmax7pays = np.reshape(capmax.values, (nb_pays_liste, nb_pays_liste))
    return qmax7pays
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

# Chargement des données pour 1 pays du code jouet et définition des capacités requises
def charge_data() :
    # Chargement des données
    df_solar = pd.read_csv('solar.csv')
    df_wind = pd.read_csv('wind_onshore.csv')
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
    pays =  pd.read_csv("data_exchange/areas.csv", header=None)
    nb_pays_liste = len(pays.values)
    capmax = pd.read_csv("data_exchange/links.csv",header=None, names=['a1','a2','links']).set_index(['a1','a2']).squeeze(axis=1)
    qmax7pays = np.reshape(capmax.values, (nb_pays_liste, nb_pays_liste))
    return qmax7pays



import numpy as np
from scipy.optimize import linprog
import pandas as pd
import numpy as np

# 1. Définition des noms de fichiers
fichiers_production = ['prodA.csv', 'prodB.csv', 'prodC.csv']
fichier_demande = 'demand2050_ademe.csv'

# --- Traitement des fichiers de production (prodA, prodB, prodC) ---

# Liste pour stocker les colonnes 'Production_Elec' de chaque fichier
liste_production_series = []

# print("Chargement des fichiers de production...")
for nom_fichier in fichiers_production:
    try:
        # Lecture du fichier CSV
        df = pd.read_csv(nom_fichier)
        
        # Sélection de la colonne 'Production_Elec' et ajout à la liste
        # On s'assure que toutes les données sont numériques (float)
        liste_production_series.append(df['Production_Elec'].astype(float))
        # print(f"  - {nom_fichier} chargé avec succès.")
    except FileNotFoundError:
        print(f"Erreur : Le fichier {nom_fichier} n'a pas été trouvé.")
        
# Concaténation des séries en une seule et conversion en np.array
# `pd.concat` va joindre les séries les unes à la suite des autres
if liste_production_series:
    production_concatenée = pd.concat(liste_production_series, ignore_index=True)
    vecteur_production = production_concatenée.to_numpy()
    
    # print("\n--- Résultat Vecteur Production (3 fichiers concaténés) ---")
    # print(f"Forme (Shape) du vecteur : {vecteur_production.shape}")
    # print(f"Type de données (Dtype) : {vecteur_production.dtype}")
    # print("Aperçu (10 premières valeurs) :")
    # print(vecteur_production[:10])
else:
    vecteur_production = np.array([])
    # print("Aucun fichier de production n'a pu être chargé. Le vecteur de production est vide.")


# --- Traitement du fichier de demande (demand2050_ademe.csv) ---

# print("\nChargement du fichier de demande...")
try:
    # Lecture du fichier CSV. Attention : il n'y a pas d'en-tête (header=None)
    # et nous devons ignorer la première colonne (l'index de la donnée)
    df_demande = pd.read_csv(fichier_demande, header=None)
    
    # Le fichier a la forme : Index, Valeur. Nous voulons la colonne des Valeurs (indice 1)
    vecteur_demande = df_demande.iloc[:, 1].astype(float).to_numpy()
    
    # print(f"  - {fichier_demande} chargé avec succès.")
    
    # print("\n--- Résultat Vecteur Demande ---")
    # print(f"Forme (Shape) du vecteur : {vecteur_demande.shape}")
    # print(f"Type de données (Dtype) : {vecteur_demande.dtype}")
    # print("Aperçu (10 premières valeurs) :")
    # print(vecteur_demande[:10])

except FileNotFoundError:
    vecteur_demande = np.array([])
    print(f"Erreur : Le fichier {fichier_demande} n'a pas été trouvé.")
    
# Les deux vecteurs numpy.array sont maintenant disponibles
# dans les variables `vecteur_production` et `vecteur_demande`.

prod_trimestriel_3_pays = vecteur_production
demand_trimestriel_3_pays = vecteur_demande[:len(prod_trimestriel_3_pays)]  # S'assurer que la demande a la même longueur que la production
qmax = 4
N = 3
T = int(len(prod_trimestriel_3_pays)/N)


# prod : vecteur avec l'entièreté des prod à chaque temps t pour chaque pays (taille = N*T)
# demand : vecteur avec l'entièreté des demand à chaque temps t pour chaque pays (taille = N*T)
# qmax : capacité maximale de flux entre deux pays
# N : nombre de pays
# T : nombre de périodes

def solve_flux(prod, demand, qmax, N):
    prod = np.array(prod)
    demand = np.array(demand)
    T = int(len(prod)/N)

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
                    row[q_index(i, j, t)] -= 1   # outflows
                    row[q_index(j, i, t)] += 1   # inflows

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
    # Initialisation de q comme un tableau 3D (T trimestres, N pays, N pays)
    q = np.zeros((T, N, N))
    idx = 0
    
    # La reconstruction est maintenant: q[t, i, j] = x[idx]
    for t in range(T):
        for j in range(N):
            for i in range(N):
                # On ne stocke que les flux inter-pays (i != j)
                if i != j:
                    # q[t, i, j] est le flux du pays i vers le pays j au trimestre t
                    q[t, i, j] = x[idx]
                    idx += 1

    r_pos = x[offset_r_pos:offset_r_pos + N*T]
    r_neg = x[offset_r_neg:offset_r_neg + N*T]

    # (optionnel) nettoyer tout petit négatif dû à la numérqiue
    eps = 1e-9
    r_pos = np.where(np.abs(r_pos) < eps, 0.0, r_pos)
    r_neg = np.where(np.abs(r_neg) < eps, 0.0, r_neg)

    return q, r_pos, r_neg



q, r_pos, r_neg = solve_flux(prod_trimestriel_3_pays, demand_trimestriel_3_pays, qmax, N)

print("Flux :\n", q)
print("r⁺ :", r_pos)
print("r⁻ :", r_neg)
print("\n")

nb_r_pos_positifs = sum(1 for val in r_pos if val > 0)
nb_r_neg_positifs = sum(1 for val in r_neg if val > 0)

nb_r_pos_positifs_A = sum(1 for val in r_pos[:T] if val > 0)
nb_r_neg_positifs_A = sum(1 for val in r_neg[:T] if val > 0)
nb_r_pos_positifs_B = sum(1 for val in r_pos[T:T*2] if val > 0)
nb_r_neg_positifs_B = sum(1 for val in r_neg[T:T*2] if val > 0)
nb_r_pos_positifs_C = sum(1 for val in r_pos[T*2:] if val > 0)
nb_r_neg_positifs_C = sum(1 for val in r_neg[T*2:] if val > 0)


print("----- Statistiques r⁺ et r⁻ -----")
print("nb heures total :", len(r_pos))
print("nb_r_pos_positifs:", nb_r_pos_positifs)
print("nb_r_neg_positifs:", nb_r_neg_positifs)
print("proportion de r negatifs", nb_r_neg_positifs / len(r_neg))
print("somme des r :", nb_r_pos_positifs + nb_r_neg_positifs )
print("nb r nuls :", len(r_pos) - (nb_r_pos_positifs + nb_r_neg_positifs))

print("---------------------------------------")


print("Pays A:")
print("nb_r_pos_positifs:", nb_r_pos_positifs_A)
print("nb_r_neg_positifs:", nb_r_neg_positifs_A)
print("proportion de r negatifs", nb_r_neg_positifs_A / len(r_neg[:T]))
print("somme des r :", nb_r_pos_positifs_A + nb_r_neg_positifs_A )
print("nb r nuls :", len(r_pos[:T]) - (nb_r_pos_positifs_A + nb_r_neg_positifs_A))

print("---------------------------------------")

print("Pays B:")
print("nb_r_pos_positifs:", nb_r_pos_positifs_B)
print("nb_r_neg_positifs:", nb_r_neg_positifs_B)
print("proportion de r negatifs", nb_r_neg_positifs_B / len(r_neg[T:T*2]))
print("somme des r :", nb_r_pos_positifs_B + nb_r_neg_positifs_B )
print("nb r nuls :", len(r_pos[T:T*2]) - (nb_r_pos_positifs_B + nb_r_neg_positifs_B))

print("---------------------------------------")

print("Pays C:")
print("nb_r_pos_positifs:", nb_r_pos_positifs_C)
print("nb_r_neg_positifs:", nb_r_neg_positifs_C)
print("proportion de r negatifs", nb_r_neg_positifs_C / len(r_neg[T*2:]))

print("somme des r :", nb_r_pos_positifs_C + nb_r_neg_positifs_C )
print("nb r nuls :", len(r_pos[T*2:]) - (nb_r_pos_positifs_C + nb_r_neg_positifs_C))




# ============================================================
# ===               GRAPHIQUES D’INTERPRÉTATION            ===
# ============================================================

import matplotlib.pyplot as plt
import numpy as np

# ---- Préparation des données ----

# r_pos et r_neg sont des vecteurs de taille N*T : on les remet en matrices T x N
r_pos_mat = r_pos.reshape(T, N)
r_neg_mat = r_neg.reshape(T, N)

# Noms des pays
pays = ["A", "B", "C"]

# ============================================================
# === 1) GRAPHIQUE r⁺ ET r⁻ POUR CHAQUE PAYS DANS LE TEMPS  ===
# ============================================================

plt.figure(figsize=(12, 6))

for i in range(N):
    plt.plot(r_pos_mat[:, i], label=f"r⁺ {pays[i]}")
    plt.plot(r_neg_mat[:, i], label=f"r⁻ {pays[i]}", linestyle="--")

plt.title("Évolution temporelle de r⁺ et r⁻")
plt.xlabel("Trimestre")
plt.ylabel("Valeur")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# === 2) HEATMAPS DES FLUX q(i→j) POUR CHAQUE TRIMESTRE     ===
# ============================================================

for t in range(T):
    plt.figure(figsize=(5, 4))
    plt.imshow(q[t], vmin=0, vmax=qmax)
    plt.colorbar(label="Flux q(i→j)")
    plt.title(f"Flux entre pays – trimestre {t}")
    plt.xlabel("Destination")
    plt.ylabel("Source")
    plt.xticks(range(N), pays)
    plt.yticks(range(N), pays)
    plt.tight_layout()
    plt.show()


# ============================================================
# === 3) FLUX TOTAUX EXPORTÉS / IMPORTÉS PAR PAYS            ===
# ============================================================

flux_export = np.sum(q, axis=(0, 2))   # somme sur t et j (sortant)
flux_import = np.sum(q, axis=(0, 1))   # somme sur t et i (entrant)

x = np.arange(N)
w = 0.3

plt.figure(figsize=(10, 5))
plt.bar(x - w, flux_export, width=w, label="Export total")
plt.bar(x + w, flux_import, width=w, label="Import total")

plt.xticks(x, pays)
plt.title("Flux total exporté / importé par pays")
plt.ylabel("Énergie transférée")
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()


# ============================================================
# === 4) PRODUCTION VS DEMANDE AVEC r⁺/r⁻ SUPERPOSÉS         ===
# ============================================================

prod_mat = prod_trimestriel_3_pays.reshape(T, N)
dem_mat = demand_trimestriel_3_pays.reshape(T, N)

for i in range(N):
    plt.figure(figsize=(10, 5))

    plt.plot(prod_mat[:, i], label="Production")
    plt.plot(dem_mat[:, i], label="Demande")
    plt.plot(r_pos_mat[:, i], label="r⁺", linestyle="--")
    plt.plot(r_neg_mat[:, i], label="r⁻", linestyle="--")

    plt.title(f"Pays {pays[i]} – Production, demande et r")
    plt.xlabel("Trimestre")
    plt.ylabel("kWh (OU unité)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


print("📊 Graphiques générés.")



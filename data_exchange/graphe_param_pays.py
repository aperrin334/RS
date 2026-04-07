# visualisation de la matrice des capacité max d'échanges
# code généré par IA qui produit un fichier image dans le dossier

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Charger les données
df = pd.read_csv('./links.csv', header=None, names=['Source', 'Destination', 'Poids'])

# Créer un graphe orienté
G = nx.DiGraph()

# Ajouter les arêtes avec leurs poids (uniquement si poids > 0)
for _, row in df.iterrows():
    source = row['Source']
    dest = row['Destination']
    poids = row['Poids']
    
    # Ajouter uniquement les arêtes avec un poids non nul
    if poids > 0:
        G.add_edge(source, dest, weight=poids)

# Créer la visualisation
plt.figure(figsize=(14, 10))

# Position des nœuds en utilisant un layout circulaire
pos = nx.circular_layout(G)

# Dessiner les nœuds
nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                       node_size=3000, alpha=0.9)

# Dessiner les labels des nœuds
nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')

# Dessiner les arêtes
nx.draw_networkx_edges(G, pos, edge_color='gray', 
                       arrows=True, arrowsize=20, 
                       arrowstyle='->', width=2,
                       connectionstyle='arc3,rad=0.1')

# Ajouter les poids sur les arêtes
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=9)

plt.title("Graphe orienté des relations entre pays", fontsize=16, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig('./graphe_pays.png', dpi=300, bbox_inches='tight')
print("Graphe sauvegardé dans graphe_pays.png")

# Afficher quelques statistiques
print("\n=== Statistiques du graphe ===")
print(f"Nombre de nœuds (pays): {G.number_of_nodes()}")
print(f"Nombre d'arêtes (relations): {G.number_of_edges()}")
print(f"\nDegré sortant (out-degree) de chaque pays:")
for node in sorted(G.nodes()):
    out_deg = G.out_degree(node)
    in_deg = G.in_degree(node)
    print(f"  {node}: sortant={out_deg}, entrant={in_deg}")

# Calculer la somme des poids sortants pour chaque pays
print(f"\nSomme des poids sortants par pays:")
for node in sorted(G.nodes()):
    total_out = sum([G[node][neighbor]['weight'] for neighbor in G.neighbors(node)])
    print(f"  {node}: {total_out}")

plt.show()

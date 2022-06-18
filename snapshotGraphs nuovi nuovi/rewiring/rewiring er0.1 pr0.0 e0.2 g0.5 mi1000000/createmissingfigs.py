import os
import networkx as nx
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.offsetbox import AnchoredText



def read_snapshot(filename):
    G = nx.Graph()
    with open(filename, 'r') as edgelistfile:
        data = csv.reader(edgelistfile)
        for row in data:
            G.add_node(int(row[0])) #superhero in first column
            G.add_node(int(row[1])) #superhero in second column
            G.add_edge(int(row[0]), int(row[1]), weight = 1)
        return G
def nclusters(data, threshold):
    data = [float(el) for el in data]
    data = sorted(data)
    start = data[0]
    max_val = start + threshold
    c = (start, max_val)
    cluster = dict()
    for i in data:
        if i <= max_val:
            if c in cluster.keys():
                cluster[c] += 1
            else:
                cluster[c] = 1
        else:
            max_val = i + threshold
            c = (i, max_val)
            cluster[c] = 1
    #ora ho il dizionario con i cluster di una run
    C_num = len(data)**2
    C_den = 0
    for k in cluster.keys():
        C_den += cluster[k]*cluster[k]
    C = C_num / C_den
    return C

def read_opinions(filename):
    colors = []
    with open(filename) as infile:
        opinions = infile.readlines()
    opinions = [float(op.strip()) for op in opinions]
    for op in opinions:
        if op < 0.4:
            colors.append("red")
        elif op >= 0.4 and op <= 0.6:
            colors.append("green")
        else:
            colors.append("blue")
    return opinions, colors

it = 1000000
G = read_snapshot(f'edgelist {it}.csv')
opinions, colors = read_opinions(f"opinions {it}.txt")
fig, ax = plt.subplots(1, 1, num=1, figsize=(10, 6))
degrees = [G.degree(n) for n in G.nodes()]
sns.histplot(x=degrees, kde=True)
plt.savefig(f"degreedist {it}.png")
plt.close()
fig, ax = plt.subplots(1, 1, num=1, figsize=(10, 8))
pos = nx.spring_layout(G, seed=1)  # positions for all nodes
nx.draw(G, pos, node_color=opinions, cmap=plt.cm.RdBu, node_size=80.0, vmin=0.0, vmax=1.0, width=0.1, alpha=1.0, ax = ax)
plt.title(f"Network at time {it}")
plt.tight_layout()
sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu, norm=plt.Normalize(vmin = 0, vmax=1))
sm._A = []
cbar = plt.colorbar(sm)
for t in cbar.ax.get_yticklabels():
    t.set_fontsize(10)
cbar.outline.set_visible(False)
cbar.ax.tick_params()
C = nclusters(opinions, 0.1)
at = AnchoredText(
    f"C={C}", prop=dict(size=15), frameon=True, loc='lower left')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax.add_artist(at)
plt.savefig(f"network {it}.png")
plt.close()

it = 100
G = read_snapshot(f'edgelist {it}.csv')
opinions, colors = read_opinions(f"opinions {it}.txt")
fig, ax = plt.subplots(1, 1, num=1, figsize=(10, 6))
degrees = [G.degree(n) for n in G.nodes()]
sns.histplot(x=degrees, kde=True)
plt.savefig(f"degreedist {it}.png")
plt.close()
fig, ax = plt.subplots(1, 1, num=1, figsize=(10, 8))
pos = nx.spring_layout(G, seed=1)  # positions for all nodes
nx.draw(G, pos, node_color=opinions, cmap=plt.cm.RdBu, node_size=80.0, vmin=0.0, vmax=1.0, width=0.1, alpha=1.0, ax = ax)
plt.title(f"Network at time {it}")
plt.tight_layout()
sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu, norm=plt.Normalize(vmin = 0, vmax=1))
sm._A = []
cbar = plt.colorbar(sm)
for t in cbar.ax.get_yticklabels():
    t.set_fontsize(10)
cbar.outline.set_visible(False)
cbar.ax.tick_params()
C = nclusters(opinions, 0.1)
at = AnchoredText(
    f"C={C}", prop=dict(size=15), frameon=True, loc='lower left')
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax.add_artist(at)
plt.savefig(f"network {it}.png")
plt.close()

            


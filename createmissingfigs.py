import os
import networkx as nx
import csv
import matplotlib.pyplot as plt
import seaborn as sns


def read_snapshot(filename):
    G = nx.Graph()
    with open(filename, 'r') as edgelistfile:
        data = csv.reader(edgelistfile)
        for row in data:
            G.add_node(int(row[0])) #superhero in first column
            G.add_node(int(row[1])) #superhero in second column
            G.add_edge(int(row[0]), int(row[1]), weight = 1)
        return G

G = read_snapshot("edgelist 1000000.csv")
fig, ax = plt.subplots(1, 1, num=1, figsize=(10, 6))
degrees = [G.degree(n) for n in G.nodes()]
sns.histplot(x=degrees, kde=True)
plt.savefig(f"degreedist 1000000.plt")
            


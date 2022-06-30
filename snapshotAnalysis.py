import matplotlib.pyplot as plt
import seaborn as sns
import os
import networkx as nx
import csv
from tqdm import tqdm
import sys
sys.path.append("/home/pansanella/mydata/GitHub/local_packages/")
sys.path.append("/data1/users/pansanella/mydata/GitHub/local_packages/")
sys.path.append("/data1/users/pansanella/mydata/GitHub/local_packages/netdspatch_local/")

import networkx as nx
import ndlib_local.ndlib.models.ModelConfig as mc
import ndlib_local.ndlib.models.opinions as op
import warnings
import tqdm
import os
import numpy as np
from matplotlib.offsetbox import AnchoredText

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

def read_snapshot(filename):
    G = nx.Graph()
    with open(filename, 'r') as edgelistfile:
        data = csv.reader(edgelistfile)
        for row in data:
            G.add_node(int(row[0])) #superhero in first column
            G.add_node(int(row[1])) #superhero in second column
            G.add_edge(int(row[0]), int(row[1]), weight = 1)
        return G

def read_opinions(filename):
    colors = []
    with open(filename) as infile:
        opinions = infile.readlines()
    opinions = [float(op.strip()) for op in opinions]
    return opinions
    

def add_opinions(graph, opinions):
    for node in graph.nodes:
        graph.nodes[node]['opinion'] = opinions[node]

    return graph

def compute_ncc(graph):
    return nx.number_connected_components(graph)

def read_snapshot(filename, it):
    G = nx.Graph()
    with open(filename, 'r') as runfile:
        its = json.load(runfile)
        edgelist = its[it]['edges']
        opinions = its[it]['status']
        for edge in edgelist:
            G.add_node(int(edge[0])) #superhero in first column
            G.add_node(int(edge[1])) #superhero in second column
            G.add_edge(int(edge[0]), int(edge[1]), weight = 1)
        nx.set_node_attributes(G, {int(k): v for k,v in opinions.items()}, name="opinion")
        return G

def compute_ncc(graph):
    return nx.number_connected_components(graph)

def compute_avgdeg(graph):
    degree = dict(graph.degree())
    s = sum(degree.values())
    return s/graph.number_of_nodes()
    
def compute_triangles(graph):
    d = dict(nx.triangles(graph))
    s = sum(d.values())
    return s/graph.number_of_nodes()

def compute_clustering(graph):
    return nx.average_clustering(graph)

def compute_nac(graph):
    return nx.degree_assortativity_coefficient(graph)

def compute_assortativity(graph):
    
    def homophily(graph, v):
        opv = graph.nodes[v]['opinion']
        degv = graph.degree[v]
        neighborsops = [graph.nodes[u]['opinion'] for u in graph.neighbors(v)]
        neighborsdeg = [graph.degree[u] for u in graph.neighbors(v)]
        E = graph.number_of_edges()
        num = 0
        den = 0
        for i in range(len(neighborsops)):
            num += (1 - ((degv)*neighborsdeg[i])/(2*E))*(opv*neighborsops[i])
            if abs(opv-neighborsops[i]) < 0.001:
                delta = 1
            else:
                delta = 0
            den += ((degv*delta)-((degv)*neighborsdeg[i])/(2*E))*(opv*neighborsops[i])              
        return num/den

    hlist = []
    
    for v in list(G.nodes()):
        hv = homophily(graph, v)
        hlist.append(hv)
    
    return sum(hlist)/len(hlist)


warnings.filterwarnings("ignore")


def save_snapshot(model, its, modelname, name, nit):
    if not os.path.exists(f'snapshotGraphs/{modelname}/{name}/'):
        os.mkdir(f'snapshotGraphs/{modelname}/{name}/')
    fig, ax = plt.subplots(1, 1, num=1, figsize=(10, 8))
    G = model.graph.graph                    
    opinions = list(its['status'].values()) 
    pos = nx.spring_layout(G, seed=1)  # positions for all nodes
    nx.draw(G, pos, node_color=opinions, cmap=plt.cm.RdBu_r, node_size=300.0, vmin=0.0, vmax=1.0, width=0.1, alpha=1.0, ax = ax)
    plt.tight_layout()
    plt.savefig(f"snapshotGraphs/{modelname}/{name}/{name}_network {nit}.png")
    plt.close()
    sns.set_style("ticks")
    fig, ax = plt.subplots(1, 1, num=1, figsize=(10, 6))
    sns.despine()
    degrees = [G.degree(n) for n in G.nodes()]
    sns.histplot(x=degrees, kde=True)
    ax.grid(True)
    ax.tick_params(direction='out', length=10, width=3, colors = "black", labelsize=30, grid_color = "black", grid_alpha = 0.1)
    ax.set_ylabel("# Nodes", size=20)
    ax.set_xlabel("Degree", size=20)
    plt.savefig(f"snapshotGraphs/{modelname}/{name}/{name}_degreedist {nit}.png")
    plt.close()
    nx.write_edgelist(G, f"snapshotGraphs/{modelname}/{name}/{name}_edgelist {nit}.csv", delimiter=",")
    with open(f"snapshotGraphs/{modelname}/{name}/{name}_opinions {nit}.txt", "w") as opfile:
        for op in list(its['status'].values()):
            opfile.write(str(op)+"\n")
    C = nclusters(opinions, 0.001)
    if C == 1.0:
        return 1


def steady_state_coevolving(model, name, nsteady=1000, max_iterations=10, sensibility = 0.00001, node_status=True, progress_bar=True):
    system_status = []
    steady_it = 0
    for it in tqdm.tqdm(range(0, max_iterations), disable=not progress_bar):
        oldgraph = model.graph.graph.copy()
        its = model.iteration(node_status)
        save_snapshots_for = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, 500, 1000, 2000, 5000, 10000, 100000, 500000, max_it]
        if it > 0:
            if np.all((its['max_diff'] < sensibility)):
                steady_it += 1
            else:
                steady_it = 0
        system_status.append(its)
        if oldgraph.edges() != model.graph.graph.edges() : steady_it = 0 
        if steady_it == nsteady:
            return system_status[:-nsteady]
        if it in save_snapshots_for:
            if save_snapshot(model, its, modelname, name, it) == 1:
                return system_status
    return system_status

modelname = "rewiring"
G = read_snapshot(f"snapshotGraphs/rewiring/rewiring er0.1 n250 pr0.0 e0.2 g0.0 mi1000000/rewiring er0.1 n250 pr0.0 e0.2 g0.0 mi1000000_edgelist initial.csv")                   
opinions = read_opinions(f"snapshotGraphs/rewiring/rewiring er0.1 n250 pr0.0 e0.2 g0.0 mi1000000/rewiring er0.1 n250 pr0.0 e0.2 g0.0 mi1000000_opinions initial.txt")
add_opinions(G, opinions)


 
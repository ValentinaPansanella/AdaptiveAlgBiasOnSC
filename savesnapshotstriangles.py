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

def homophily(graph, v):
    neighbors = list(graph.neighbors(v))
    if len(neighbors) > 0:
        neighborsops = [graph.nodes[u]['opinion'] for u in neighbors]
        h =  sum([(((-2/max(graph.nodes[v]['opinion'], 1-graph.nodes[v]['opinion']))*(abs(graph.nodes[v]['opinion']-op)))+1) for op in neighborsops])/len(neighborsops)
        return h
    else:
        return 0

warnings.filterwarnings("ignore")

def save_first_snapshot(model, initial_status, modelname, name):
    if not os.path.exists(f'snapshotGraphs/{modelname}/{name}/'):
        os.mkdir(f'snapshotGraphs/{modelname}/{name}/')
    fig, ax = plt.subplots(1, 1, num=1, figsize=(10, 8))
    G = model.graph.graph         
    opinions = initial_status
    G = add_opinions(G, opinions)
    pos = nx.spring_layout(G, seed=1)  # positions for all nodes
    nx.draw(G, pos, node_color=opinions, cmap=plt.cm.RdBu_r, node_size=300.0, vmin=0.0, vmax=1.0, width=0.1, alpha=1.0, ax = ax)
    plt.tight_layout()
    plt.savefig(f"snapshotGraphs/{modelname}/{name}/{name}_network initial.png")
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
    plt.savefig(f"snapshotGraphs/{modelname}/{name}/{name}_degreedist initial.png")
    plt.close()
    nx.write_edgelist(G, f"snapshotGraphs/{modelname}/{name}/{name}_edgelist initial.csv", delimiter=",")
    with open(f"snapshotGraphs/{modelname}/{name}/{name}_opinions initial.txt", "w") as opfile:
        for op in initial_status:
            opfile.write(str(op)+"\n")    

    hlist = []
    for v in list(G.nodes()):
        hv = homophily(graph, v)
        hlist.append(hv)

    with open(f"snapshotGraphs/{modelname}/{name}/{name}_homophily initial.txt", "w") as hfile:
        for h in hlist:
            hfile.write(str(h)+"\n")


def save_snapshot(model, its, modelname, name, nit):
    if not os.path.exists(f'snapshotGraphs/{modelname}/{name}/'):
        os.mkdir(f'snapshotGraphs/{modelname}/{name}/')
    fig, ax = plt.subplots(1, 1, num=1, figsize=(10, 8))
    G = model.graph.graph                    
    opinions = list(its['status'].values()) 
    G = add_opinions(G, opinions)
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
    hlist = []
    for v in list(G.nodes()):
        hv = homophily(graph, v)
        hlist.append(hv)

    with open(f"snapshotGraphs/{modelname}/{name}/{name}_homophily {nit}.txt", "w") as hfile:
        for h in hlist:
            hfile.write(str(h)+"\n")

def save_last_snapshot(model, its, modelname, name, nit):
    if not os.path.exists(f'snapshotGraphs/{modelname}/{name}/'):
        os.mkdir(f'snapshotGraphs/{modelname}/{name}/')
    fig, ax = plt.subplots(1, 1, num=1, figsize=(10, 8))
    G = model.graph.graph                    
    opinions = list(its['status'].values()) 
    G = add_opinions(G, opinions)
    pos = nx.spring_layout(G, seed=1)  # positions for all nodes
    nx.draw(G, pos, node_color=opinions, cmap=plt.cm.RdBu_r, node_size=300.0, vmin=0.0, vmax=1.0, width=0.1, alpha=1.0, ax = ax)
    plt.tight_layout()
    plt.savefig(f"snapshotGraphs/{modelname}/{name}/{name}_network {nit} last.png")
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
    plt.savefig(f"snapshotGraphs/{modelname}/{name}/{name}_degreedist {nit} last.png")
    plt.close()
    nx.write_edgelist(G, f"snapshotGraphs/{modelname}/{name}/{name}_edgelist {nit} last.csv", delimiter=",")
    with open(f"snapshotGraphs/{modelname}/{name}/{name}_opinions {nit} last.txt", "w") as opfile:
        for op in list(its['status'].values()):
            opfile.write(str(op)+"\n")
    C = nclusters(opinions, 0.001)
    hlist = []
    
    for v in list(G.nodes()):
        hv = homophily(graph, v)
        hlist.append(hv)

    with open(f"snapshotGraphs/{modelname}/{name}/{name}_homophily {nit} last.txt", "w") as hfile:
        for h in hlist:
            hfile.write(str(h)+"\n")


def steady_state_coevolving(model, name, initial_status, start, nsteady=1000, max_iterations=10, sensibility = 0.00001, node_status=True, progress_bar=True):
    system_status = []
    steady_it = 0
      
    for it in tqdm.tqdm(range(start, max_iterations), disable=not progress_bar):
        
        if it == 0:
            save_first_snapshot(model, initial_status, modelname, name)
        
        oldgraph = model.graph.graph.copy()
        
        its = model.iteration(node_status)

        save_snapshots_for = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, 500, 1000, 2000, 5000, 10000, 100000, 500000, max_it]
        if it in save_snapshots_for:
            C = save_snapshot(model, its, modelname, name, it)
            if C == 1.0:
                return system_status

        if it > start:
            if np.all((its['max_diff'] < sensibility)):
                steady_it += 1
            else:
                steady_it = 0

        system_status.append(its)

        if oldgraph.edges() != model.graph.graph.edges() : steady_it = 0 

        if steady_it == nsteady:
            save_last_snapshot(model, its, modelname, name, it)
            return system_status[:-nsteady]

    return system_status

modelname = "triangles rewiring"
n = 250
max_it = 1000000
initial_status = np.random.random_sample(n)
graphname = 'er'
p = 0.1
i = 0
start = 0
graph = nx.erdos_renyi_graph(n, p, seed = i)
ncc = nx.number_connected_components(graph) 
print('there are {} ncc'.format(ncc))
while ncc > 1:
    i+=1

for pr in [0.0, 0.5]:
    for e in [0.2]:
        for g in [0.0, 0.5]:
            
            name = f"{modelname} {graphname}{p} n{n} pr{pr} e{e} g{g} mi{max_it}"
            graph = nx.erdos_renyi_graph(n, p, seed = i)

            if os.path.exists(f"snapshotGraphs/{modelname}/{name}/"):
                save_snapshots_for = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, 500, 1000, 2000, 5000, 10000, 100000, 500000, max_it]
                for nit in reversed(save_snapshots_for):
                    try:
                        graph = read_snapshot(f"snapshotGraphs/{modelname}/{name}/{name}_edgelist {nit}.csv")
                        initial_status = read_opinions(f"snapshotGraphs/{modelname}/{name}/{name}_opinions {nit}.txt")
                        start = nit
                        break
                    except:
                        continue
            else:
                nit = start = 0
            
            print("starting from ", start)

            if modelname == "rewiring":
                model = op.AdaptiveAlgorithmicBiasModel(graph)
            else:
                model = op.AdaptivePeerPressureAlgorithmicBiasModel(graph)

            config = mc.Configuration()
            config.add_model_parameter("epsilon", e)
            config.add_model_parameter("gamma", g)
            config.add_model_parameter("p", pr)
            model.set_initial_status(config, initial_status)
            status = steady_state_coevolving(model, name, initial_status, start, max_iterations=max_it+1, node_status=True, progress_bar=True)                
 

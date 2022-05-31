import matplotlib.pyplot as plt
import seaborn as sns
import os
import networkx as nx
import csv
from tqdm import tqdm
import ndlib_local.ndlib.models.ModelConfig as mc
import ndlib_local.ndlib.models.opinions as op
import warnings
import tqdm
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
    for op in opinions:
        if op < 0.4:
            colors.append("red")
        elif op >= 0.4 and op <= 0.6:
            colors.append("green")
        else:
            colors.append("blue")
    return opinions, colors
    

def add_opinions(graph, opinions):
    for node in graph.nodes:
        graph.nodes[node]['opinion'] = opinions[node]

    return graph

def compute_ncc(graph):
    return nx.number_connected_components(graph)

def iteration_bunch(model, name, niterations=10, node_status=True, progress_bar=True):
    if not os.path.exists(f"snapshotGraphs nuovi nuovi/{modelname}/{name}/"):
        os.mkdir(f"snapshotGraphs nuovi nuovi/{modelname}/{name}/")
    system_status = []
    params = name.split(' ')
    for it in tqdm.tqdm(range(0, niterations), disable=not progress_bar):    
        its = model.iteration(node_status)
        system_status.append(its)
        save_snapshots_for = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, 500, 1000, 2000, 5000, 10000, 100000, 500000, max_it]
        if it in save_snapshots_for:
            fig, ax = plt.subplots(1, 1, num=1, figsize=(10, 8), dpi=600)
            G = model.graph.graph
            pos = nx.spring_layout(G, seed=1)  # positions for all nodes
            nx.draw(G, pos, node_color=list(its['status'].values()), cmap=plt.cm.RdBu, node_size=80.0, vmin=0.0, vmax=1.0, width=0.1, alpha=1.0, ax = ax)
            plt.title(f"Network at time {it}")
            plt.tight_layout()
            sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu, norm=plt.Normalize(vmin = 0, vmax=1))
            sm._A = []
            cbar = plt.colorbar(sm)
            for t in cbar.ax.get_yticklabels():
                t.set_fontsize(10)
            cbar.outline.set_visible(False)
            cbar.ax.tick_params()
            opinions = list(its['status'].values())
            C = nclusters(opinions, 0.1)
            at = AnchoredText(
                f"C={C}", prop=dict(size=15), frameon=True, loc='lower left')
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(at)
            plt.savefig(f"snapshotGraphs nuovi nuovi/{modelname}/{name}/network {it}.png")
            plt.close()
            fig, ax = plt.subplots(1, 1, num=1, figsize=(10, 6), dpi=600)
            degrees = [G.degree(n) for n in G.nodes()]
            sns.histplot(x=degrees, bins=30, binwidth=1, kde=True)
            plt.title(f"Degree distribution at time {it}")
            plt.xlabel("Degree")
            plt.ylabel("# Nodes")
            plt.tight_layout()
            plt.savefig(f"snapshotGraphs nuovi nuovi/{modelname}/{name}/degreedist {it}.png")
            plt.close()

            nx.write_edgelist(G, f"snapshotGraphs nuovi nuovi/{modelname}/{name}/edgelist {it}.csv", delimiter=",")
            with open(f"snapshotGraphs nuovi nuovi/{modelname}/{name}/opinions {it}.txt", "w") as opfile:
                for op in list(np.array(list(its['status'].values()))):
                    opfile.write(str(op)+"\n")
            
            if C <= 1.0:
                break
        
        # return system_status

modelname = "triangles rewiring"

if not os.path.exists(f"snapshotGraphs nuovi nuovi/{modelname}/"):
    os.mkdir(f"snapshotGraphs nuovi nuovi/{modelname}/")

n = 250
max_it = 1000000

for graphname in ['er', 'ba']:
    if graphname == 'er':
        p = 0.1
        graph = nx.erdos_renyi_graph(n, p, seed=0)
    else:
        p = 5
        graph = nx.barabasi_albert_graph(n, p, seed=0)
    for pr in [0.0, 0.5]:
        for e in [0.2]:
            for g in [0.0,0.5,1.0,1.5]:
                name = f"{modelname} {graphname}{p} pr{pr} e{e} g{g} mi{max_it}"
                print(name)
                if not os.path.exists(f"snapshotGraphs nuovi nuovi/{modelname}/{name}/"):
                    if modelname == "rewiring":
                        model = op.AdaptiveAlgorithmicBiasModel(graph)
                    else:
                        model = op.AdaptivePeerPressureAlgorithmicBiasModel(graph)
                    config = mc.Configuration()
                    config.add_model_parameter("epsilon", e)
                    config.add_model_parameter("gamma", g)
                    config.add_model_parameter("p", pr)
                    model.set_initial_status(config)
                    status = iteration_bunch(model, name, niterations=max_it+1, node_status=True, progress_bar=True) 
                else:
                    start = 0
                    for filename in os.listdir(f"snapshotGraphs nuovi nuovi/{modelname}/{name}/"):
                        if filename.startswith("edgelist"):
                            tmp = int(filename.split(' ')[1].split('.')[0])
                            if tmp > start:
                                start = tmp
                    if start < max_it:
                        gr = read_snapshot(f"snapshotGraphs nuovi nuovi/{modelname}/{name}/edgelist {start}.csv")
                        opinions, colors = read_opinions(f"snapshotGraphs nuovi nuovi/{modelname}/{name}/opinions {start}.txt")
                        if modelname == "rewiring":
                            model = op.AdaptiveAlgorithmicBiasModel(gr)
                        else:
                            model = op.AdaptivePeerPressureAlgorithmicBiasModel(gr)
                        config = mc.Configuration()
                        config.add_model_parameter("epsilon", e)
                        config.add_model_parameter("gamma", g)
                        config.add_model_parameter("p", pr)
                        model.set_initial_status(config, initial_status=opinions)
                        status = iteration_bunch(model, name, niterations=max_it+1, node_status=True, progress_bar=True) 
                    else:
                        print("everything's already there!!")
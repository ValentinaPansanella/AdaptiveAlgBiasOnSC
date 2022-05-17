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
warnings.filterwarnings("ignore")

def steady_state_coevolving(model, name, max_iterations=100000, nsteady=1000, sensibility=0.00001, node_status=True, progress_bar=False):
    if not os.path.exists(f"snapshotGraphs/edgelist {name} 0.csv"):
        print(type(model.graph.graph))
        system_status = []
        steady_it = 0
        for it in tqdm.tqdm(range(0, max_iterations), disable=not progress_bar):            
            its = model.iteration(node_status)
            
            if it == 0:
                G = model.graph.graph
                nx.write_edgelist(G, f"snapshotGraphs/edgelist {name} {it}.csv", delimiter=",")
                with open(f"snapshotGraphs/opinions {name} {it}.txt", "w") as opfile:
                    first_values = list(its['status'].values())
                    for op in first_values:
                        opfile.write(str(op)+"\n")

            elif it > 0:
                old = np.array(list(system_status[-1]['status'].values()))
                actual = np.array(list(its['status'].values()))
                # res = np.abs(old - actual)
                if np.all((its['max_diff'] < sensibility)):
                    steady_it += 1
                else:
                    steady_it = 0

                if it % 500 == 0:
                    G = model.graph.graph
                    nx.write_edgelist(G, f"snapshotGraphs/edgelist {name} {it}.csv", delimiter=",")
                    with open(f"snapshotGraphs/opinions {name} {it}.txt", "w") as opfile:
                        for op in list(actual):
                            opfile.write(str(op)+"\n")

            system_status.append(its)
            
            if steady_it == nsteady:
                G = model.graph.graph
                nx.write_edgelist(G, f"snapshotGraphs/edgelist {name} {it}.csv", delimiter=",")
                with open(f"snapshotGraphs/opinions {name} {it}.txt", "w") as opfile:
                    for op in list(actual):
                        opfile.write(str(op)+"\n")
                        
                return system_status[:-nsteady]

        return system_status


n = 250
max_it = 100000

for graphname in ['er', 'ba']:
    if graphname == 'er':
        p = 0.1
        graph = nx.erdos_renyi_graph(n, p)
    else:
        p = 5
        graph = nx.barabasi_albert_graph(n, p)
    for pr in [0.5]:
        for e in [0.2, 0.3, 0.4]:
            for g in [0.0, 0.5, 1.0, 1.5]:
                name = f"rewiring {graphname}{p} pr{pr} e{e} g{g} mi{max_it}"
                model = op.AdaptiveAlgorithmicBiasModel(graph)
                config = mc.Configuration()
                config.add_model_parameter("epsilon", e)
                config.add_model_parameter("gamma", g)
                config.add_model_parameter("p", pr)
                model.set_initial_status(config)
                steady_status = steady_state_coevolving(model=model, name=name, max_iterations=max_it, nsteady=1000, sensibility=0.00001, node_status=True, progress_bar=True)                                
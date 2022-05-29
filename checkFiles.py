import json 
import os
import networkx as nx

def checkfiles(modelname, graphname):
    n = 250
    if graphname == "ba":
        p = 5
        graph = nx.barabasi_albert_graph(n, p)
    else:
        p = 0.1
        graph = nx.erdos_renyi_graph(n, p)
    if modelname == "triangles rewiring":
        nruns = 10
    else:
        nruns = 30
    max_it = 100000
    for pr in [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
        for e in [0.4, 0.3, 0.2]:
            for g in [0.0, 0.4, 0.8, 1.2, 1.6]:
                final_opinions = dict()
                final_iterations = dict()
                name = f"{modelname} {graphname}{p} pr{pr} e{e} g{g} mi{max_it}"
                #performing multiple runs with same parameters for statistically valid results
                if os.path.exists(f"aggregate/final_opinions {name}.json"):
                    with open(f"aggregate/final_opinions {name}.json") as fo:
                        final_opinions = json.load(fo)
                    with open(f"aggregate/final_iterations {name}.json") as fi:
                        final_iterations = json.load(fi)
                    dict_keys = [int(el) for el in final_opinions.keys()]
                    max_key = max(dict_keys)+1
                else:
                    final_opinions = dict()
                    final_iterations = dict()
                    max_key = 0
                    
                dict_keys_fo = list(final_opinions.keys())
                dict_keys_fi = list(final_iterations.keys())

                if len(dict_keys_fo) == len(dict_keys_fi) >= nruns-1:
                    continue
                else:
                    print(max_key)
                    print(nruns - max_key, 'runs missing from dict')
                    print('for ', name)
                    continue

checkfiles("triangles rewiring", "ba")
                
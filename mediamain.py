import sys
sys.path.append("/../../GitHub/local_packages/")
sys.path.append("/../../GitHub/local_packages/")
sys.path.append("/../../GitHub/local_packages/netdspatch_local/")

import json
import os
import networkx as nx
import ndlib_local.ndlib.models.ModelConfig as mc
import ndlib_local.ndlib.models.opinions as op
import warnings
warnings.filterwarnings("ignore")

def multiple_exec():
    graphname = "er"
    n = 100
    graph = nx.complete_graph(n)
    nruns = 1
    max_it = 1000000
    
    i = 0
    for mo in [[0.05, 0.95]]:
        # titles = ['extremist', 'moderate', 'polarised', 'balanced']
        titles = ['polarised']
        for pm in [0.0, 0.5, 1.0]:
            for e in [0.32]:
                for g in [0.0]:
                    # final_opinions = dict()
                    # final_iterations = dict()
                    name = f"media {titles[i]} complete pm{pm} e{e} g{g} mi{max_it}"
                    #performing multiple runs with same parameters for statistically valid results
                    # if os.path.exists(f"aggregate/final_opinions {name}.json"):
                    #     print(f"dictionary already exists for {name}")
                    #     with open(f"aggregate/final_opinions {name}.json") as fo:
                    #         final_opinions = json.load(fo)
                    #     with open(f"aggregate/final_iterations {name}.json") as fi:
                    #         final_iterations = json.load(fi)
                    #     dict_keys = [int(el) for el in final_opinions.keys()]
                    #     max_key = max(dict_keys)+1
                    #     print(max_key)
                    # else:
                    #     final_opinions = dict()
                    #     final_iterations = dict()
                    #     max_key = 0

                max_key = 0

                for nr in (range(max_key, nruns)):
                    print(f"doing {name} run {nr}")
                    # Model selection
                    model = op.AlgorithmicBiasMediaModel(graph)
                    # Model configuration
                    config = mc.Configuration()
                    config.add_model_parameter("epsilon", e)
                    config.add_model_parameter("gamma", g)
                    config.add_model_parameter("gamma_media", g)
                    config.add_model_parameter("k", len(mo))
                    config.add_model_parameter("p", pm)
                    model.set_initial_status(config)
                    # # Simulation execution
                    steady_status = model.steady_state(max_iterations=max_it, nsteady=1000, sensibility=0.00001, node_status=True, progress_bar=True)
                    
                    with open(f"res/media/for_spaghetti/{name} nr{nr}.json", "w") as jsonfile:
                        json.dump(steady_status, jsonfile)

                    # last_opinions = [v for k, v in steady_status[len(steady_status)-1]['status'].items()]
                    # n_its = int(steady_status[len(steady_status)-1]['iteration'])
                        
                    # final_opinions[nr] = last_opinions
                    # final_iterations[nr] = n_its

                # with open(f"aggregate/final_opinions {name}.json", "w") as f:
                #     json.dump(final_opinions, f)

                # with open(f"aggregate/final_iterations {name}.json", "w") as f:
                #     json.dump(final_iterations, f)

                # with open(f"aggregate/final_opinions {name}.json") as fo:
                #         final_opinions = json.load(fo)
                        
                # with open(f"aggregate/final_iterations {name}.json") as fi:
                #     final_iterations = json.load(fi)
                    
                # dict_keys_fo = list(final_opinions.keys())
                # dict_keys_fi = list(final_iterations.keys())

                # if len(dict_keys_fo) == len(dict_keys_fi) >= nruns-1:
                #     print("ok")
                #     continue
                # else:
                #     return
        i += 1

multiple_exec()

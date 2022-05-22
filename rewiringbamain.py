import sys
sys.path.append("/home/pansanella/mydata/GitHub/local_packages/")
sys.path.append("/data1/users/pansanella/mydata/GitHub/local_packages/")
sys.path.append("/data1/users/pansanella/mydata/GitHub/local_packages/netdspatch_local/")

import json
import os
import networkx as nx
import ndlib_local.ndlib.models.ModelConfig as mc
import ndlib_local.ndlib.models.opinions as op
import warnings
warnings.filterwarnings("ignore")

graph = nx.barabasi_albert_graph(250, 5)
nruns = 30
max_it = 100000

for pr in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for e in [0.2, 0.3, 0.4]:
        for g in [0.0, 0.4, 0.8, 1.2, 1.6]:
            name = f"rewiring ba5 pr{pr} e{e} g{g} mi{max_it}"
            #performing multiple runs with same parameters for statistically valid results
            if os.path.exists(f"aggregate/final_opinions {name}.json"):
                with open(f"aggregate/final_opinions {name}.json") as fo:
                    final_opinions = json.load(fo)
                with open(f"aggregate/final_iterations {name}.json") as fi:
                    final_iterations = json.load(fi)
                dict_keys = list(final_opinions.keys())
                max_key = int(max(dict_keys)) + 1
            else:
                final_opinions = dict()
                final_iterations = dict()
                max_key = 0

            for nr in (range(max_key, nruns)):
                print(f"doing {name} run {nr}")
                # Model selection
                model = op.AdaptiveAlgorithmicBiasModel(graph)
                # Model configuration
                config = mc.Configuration()
                config.add_model_parameter("epsilon", e)
                config.add_model_parameter("gamma", g)
                config.add_model_parameter("p", pr)
                model.set_initial_status(config)
                # # Simulation execution
                steady_status = model.steady_state(max_iterations=max_it, nsteady=1000, sensibility=0.00001, node_status=True, progress_bar=True)
                last_opinions = [v for k, v in steady_status[len(steady_status)-1]['status'].items()]
                n_its = int(steady_status[len(steady_status)-1]['iteration'])
                    
                final_opinions[nr] = last_opinions
                final_iterations[nr] = n_its

            with open(f"aggregate/final_opinions {name}.json", "w") as f:
                json.dump(final_opinions, f)

            with open(f"aggregate/final_iterations {name}.json", "w") as f:
                json.dump(final_iterations, f)
            
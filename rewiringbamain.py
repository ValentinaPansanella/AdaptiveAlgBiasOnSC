import networkx as nx
import ndlib_local.ndlib.models.ModelConfig as mc
import ndlib_local.ndlib.models.opinions as op
import warnings
warnings.filterwarnings("ignore")

graph = nx.barabasi_albert_graph(250, 5)
nruns = 5
max_it = 100000

for pr in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for e in [0.2, 0.3, 0.4]:
        for g in [0.0, 0.4, 0.8, 1.2, 1.6]:
            final_opinions = dict()
            final_iterations = dict()
            name = f"rewiring ba5 pr{pr} e{e} g{g} mi{max_it}"
            #performing multiple runs with same parameters for statistically valid results
            # if not os.path.exists(f"finals/final_opinions {name}.json"):
            for nr in (range(nruns)):
                print(f"doing {name} run {nr}")
                # Model selection
                # model = op.AdaptiveAlgorithmicBiasModel(graph)
                model = op.AdaptiveAlgorithmicBiasModel(graph)
                # Model configuration
                config = mc.Configuration()
                config.add_model_parameter("epsilon", e)
                config.add_model_parameter("gamma", g)
                config.add_model_parameter("p", pr)
                model.set_initial_status(config)
                # # Simulation execution
                steady_status = model.steady_state(max_iterations=max_it, nsteady=1000, sensibility=0.00001, node_status=True, progress_bar=True)
        #         last_opinions = [v for k, v in steady_status[len(steady_status)-1]['status'].items()]
        #         n_its = int(steady_status[len(steady_status)-1]['iteration'])
                    
            #         final_opinions[nr] = last_opinions
            #         final_iterations[nr] = n_its

            #     with open(f"finals/final_opinions {name}.json", "w") as f:
            #         json.dump(final_opinions, f)

            #     with open(f"finals/final_iterations {name}.json", "w") as f:
            #         json.dump(final_iterations, f)
            # else: 
            #     print(f"{name} already present: skipping")
            #     continue

                
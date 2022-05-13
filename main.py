path_to_local_libs = 'C:\\Users\\valen\\Documents\\GitHub\\Lib\\'
import networkx as nx
try:
    import ndlib_local.ndlib.models.ModelConfig as mc
    import ndlib_local.ndlib.models.opinions as op
except ModuleNotFoundError:
    import sys
    sys.path.insert(0,path_to_local_libs)
    import ndlib_local.ndlib.models.ModelConfig as mc
    import ndlib_local.ndlib.models.opinions as op
import warnings
import json
import os.path
warnings.filterwarnings("ignore")

#Adaptive Algorithmic Bias Model

graph = nx.complete_graph(100) 
max_it = 100000 #maximum number of iterations for run

pr = 0.1 #rewiring probability
e = 0.32 #confidence bound
g = 1.0 #algorithmic bias

name = f"rewiring complete pr{pr} e{e} g{g} mi{max_it}"

model = op.AdaptiveAlgorithmicBiasModel(graph)
# Model configuration
config = mc.Configuration()
config.add_model_parameter("epsilon", e)
config.add_model_parameter("gamma", g)
config.add_model_parameter("p", pr)
model.set_initial_status(config)
# # Simulation execution
steady_status = model.steady_state(max_iterations=max_it, nsteady=1000, sensibility=0.00001, node_status=True, progress_bar=True)

graph = nx.complete_graph(100) 
max_it = 100000 #maximum number of iterations for run

pr = 0.1 #rewiring probability
e = 0.32 #confidence bound
g = 1.0 #algorithmic bias

####################################################################################################################################
#Adaptive Algorithmic Bias Model with peer pressure

name = f"rewiring complete pr{pr} e{e} g{g} mi{max_it}"

model = op.AdaptivePeerPressureAlgorithmicBiasModel(graph)
# Model configuration
config = mc.Configuration()
config.add_model_parameter("epsilon", e)
config.add_model_parameter("gamma", g)
config.add_model_parameter("p", pr)
model.set_initial_status(config)
# # Simulation execution
steady_status = model.steady_state(max_iterations=max_it, nsteady=1000, sensibility=0.00001, node_status=True, progress_bar=True)


                
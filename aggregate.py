import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

path = "finals/"

graph = "er"
n = 250
p = 0.1
nruns = 10
max_it = 100000

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

def finaldistribution(name):
    sns.set_style("whitegrid")
    jsonfile = open(f'finals/final_opinions {name}.json')
    data = json.load(jsonfile)
    for nr in data.keys():
        finalops = list(data[str(nr)])
        x = [i for i in range(250)]
        y = sorted(finalops)    
        node2col = {}
        for node in x:
            if y[node] < 0.33:
                node2col[node] = '#3776ab'
            elif 0.33 <= y[node] <= 0.66:
                node2col[node] = '#FFA500'
            else:
                node2col[node] = '#FF0000'
        for node in x:
            plt.scatter(x[node], y[node], s = 0.2, c = node2col[node])
        plt.ylim(-0.01, 1.01)
        plt.savefig(f"plots/final_opinions {name} nr{nr}.png")
        plt.tight_layout()
        plt.close()

# with open(f"finals/aggregate rewiring {graph}{p}.csv", "w") as ofile:
#     for pr in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
#         for e in [0.2, 0.3, 0.4]:
#             for g in [0.0, 0.4, 0.8, 1.2, 1.6]:
#                 name = f"rewiring {graph}{p} pr{pr} e{e} g{g} mi{max_it}"
#                 try:
#                     jsonfile = open(f'finals/final_opinions {name}.json')
#                     data = json.load(jsonfile)
#                     ncarray = []
#                     avgfoarray = []
#                     for nr in data.keys():
#                         finalops = list(data[nr])
#                         nc = nclusters(finalops, 0.01)
#                         fo = np.average(np.array(finalops))
#                         ncarray.append(nc)
#                         avgfoarray.append(fo)
#                     ncarray = np.array(ncarray)
#                     avg_nc = np.average(ncarray)
#                     std_nc = np.std(ncarray)
#                     avgfoarray = np.array(avgfoarray)
#                     avg_fo = np.average(avgfoarray)
#                     std_fo = np.std(avgfoarray)
#                     towrite = f"rewiring,{graph},{p},{n},{nruns},{pr},{e},{g},{max_it},{avg_nc},{std_nc},{avg_fo},{std_fo}\n"
#                     ofile.write(towrite)
#                 except FileNotFoundError:
#                     continue

for pr in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    for e in [0.2, 0.3, 0.4]:
        for g in [0.0, 0.4, 0.8, 1.2, 1.6]:
            name = f"rewiring {graph}{p} pr{pr} e{e} g{g} mi{max_it}"
            print(f"plotting {name}")
            finaldistribution(name)
 
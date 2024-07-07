import pickle
from collections import Counter
import igraph as ig
import jraph
import numpy as np
import sys
sys.path.append("/home/chenhaojun/DIffUCO")
from DatasetCreator.Gurobi import GurobiSolver
from tqdm import tqdm
from pathlib import Path
import shutil
import argparse

def from_igraph_to_jgraph(igraph, zero_edges = True, double_edges = True, _np = np):
    num_vertices = igraph.vcount()
    edge_arr = _np.array(igraph.get_edgelist())
    if(double_edges):
        print("ATTENTION: edges will be dublicated in this method!")
        if(igraph.ecount() > 0):
            undir_receivers = edge_arr[:, 0]
            undir_senders = edge_arr[:, 1]
            receivers = _np.concatenate([undir_receivers[:, np.newaxis], undir_senders[:, np.newaxis]], axis=-1)
            receivers = _np.ravel(receivers)
            senders = _np.concatenate([undir_senders[:, np.newaxis], undir_receivers[:, np.newaxis]], axis=-1)
            senders = _np.ravel(senders)
            edges =  _np.ones((senders.shape[0], 1))
        else:
            receivers = _np.ones((0,), dtype = np.int32)
            senders = _np.ones((0,), dtype = np.int32)
            edges =  _np.ones((0, 1))

        if (not zero_edges):
            edge_weights = igraph.es["weight"]
            edges = _np.concatenate([edge_weights, edge_weights], axis=0)
    else:
        if(igraph.ecount() > 0):
            senders = edge_arr[:, 0]
            receivers = edge_arr[:, 1]
            edges =  _np.ones((senders.shape[0], 1))
        else:
            receivers = _np.ones((0,), dtype = np.int32)
            senders = _np.ones((0,), dtype = np.int32)
            edges =  _np.ones((0, 1))

        if (not zero_edges):
            edge_weights = igraph.es["weight"]
            edges = _np.array(edge_weights)

    nodes = _np.zeros((num_vertices, 1))
    globals = _np.array([num_vertices])
    n_node = _np.array([num_vertices])
    n_edge = _np.array([receivers.shape[0]])

    jgraph = jraph.GraphsTuple(senders = senders, receivers = receivers, edges = edges, nodes = nodes, n_edge = n_edge , n_node = n_node, globals = globals )
    return jgraph

def solve_graph(H_graph, g, gurobi_solve=True):
    """
    Solve the graph instance for the dataset using gurobi if self.gurobi_solve is True, otherwise return None Tuple

    :param H_graph: jraph graph instance
    :param g: igraph graph instance
    :return: (Energy, boundEnergy, solution, runtime, H_graph_compl)
    """
    if gurobi_solve:
        H_graph_compl = from_igraph_to_jgraph(g, double_edges=False)
        _, Energy, solution, runtime = GurobiSolver.solveMIS_as_MIP(H_graph)
        return Energy, None, solution, runtime, H_graph_compl

    else:
        # in case gurobi is not used, arbitrary values are returned and for MaxCl, the complement graph is returned
        Energy = 0.
        boundEnergy = 0.
        solution = np.ones_like(H_graph.nodes)
        runtime = None

        H_graph_compl = from_igraph_to_jgraph(g, double_edges=False)

        return Energy, boundEnergy, solution, runtime, H_graph_compl

def igraph_to_jraph(g: ig.Graph):
    """
    Convert igraph graph to jraph graph

    :param g: igraph graph
    :return: (H_graph, density, graph_size)
    """
    density = 2 * g.ecount() / (g.vcount() * (g.vcount() - 1))
    graph_size = g.vcount()
    return from_igraph_to_jgraph(g), density, graph_size

parser = argparse.ArgumentParser()
parser.add_argument('--sub_graph_group', default="/home/chenhaojun/DIffUCO/Data/KS_3_split/folder_2", type = str, help='licence base path')
# sub_graph_group = "/home/chenhaojun/DIffUCO/Data/KS_3_split/folder_1"
args = parser.parse_args()
sub_graph_group = args.sub_graph_group
print(f"dealing with {sub_graph_group} ...")

sub_graph_group = Path(sub_graph_group)
save_graph_dir = sub_graph_group / "solution"
save_graph_dir.mkdir(parents=True, exist_ok=True)
# for idx, sub_graph in tqdm(enumerate(sub_graph_group.iterdir())):
#     with open(str(sub_graph), 'rb') as f:
#         graph = pickle.load(f)
#         edges = Counter(graph['overlap_id'])
#         g = ig.Graph([(edge[0], edge[1]) for edge in edges])

#         H_graph, density, graph_size = igraph_to_jraph(g)
#         Energy, boundEnergy, solution, runtime, compl_H_graph = solve_graph(H_graph, g)

#         indexed_solution_dict = {}

#         indexed_solution_dict["Energies"] = Energy
#         indexed_solution_dict["H_graphs"] = H_graph
#         indexed_solution_dict["gs_bins"] = solution
#         indexed_solution_dict["graph_sizes"] = graph_size
#         indexed_solution_dict["densities"] = density
#         indexed_solution_dict["runtimes"] = runtime
#         indexed_solution_dict["upperBoundEnergies"] = boundEnergy
#         indexed_solution_dict["compl_H_graphs"] = compl_H_graph
#         indexed_solution_dict["coordinate"] = graph['unit_vector']

#         save_path = save_graph_dir / f"solution_{idx}.pickle"
#         with open(str(save_path), 'wb') as f:
#             pickle.dump(indexed_solution_dict, f)

total_files = len(list(sub_graph_group.iterdir()))

for idx, sub_graph in enumerate(tqdm(sub_graph_group.iterdir(), total=total_files, desc="Processing sub-graphs")):
    with open(str(sub_graph), 'rb') as f:
        graph = pickle.load(f)
        edges = Counter(graph['overlap_id'])
        g = ig.Graph([(edge[0], edge[1]) for edge in edges])

        H_graph, density, graph_size = igraph_to_jraph(g)
        Energy, boundEnergy, solution, runtime, compl_H_graph = solve_graph(H_graph, g)

        indexed_solution_dict = {}

        indexed_solution_dict["Energies"] = Energy
        indexed_solution_dict["H_graphs"] = H_graph
        indexed_solution_dict["gs_bins"] = solution
        indexed_solution_dict["graph_sizes"] = graph_size
        indexed_solution_dict["densities"] = density
        indexed_solution_dict["runtimes"] = runtime
        indexed_solution_dict["upperBoundEnergies"] = boundEnergy
        indexed_solution_dict["compl_H_graphs"] = compl_H_graph
        indexed_solution_dict["coordinate"] = graph['unit_vector']

        save_path = save_graph_dir / f"solution_{idx}.pickle"
        with open(str(save_path), 'wb') as f:
            pickle.dump(indexed_solution_dict, f)

        


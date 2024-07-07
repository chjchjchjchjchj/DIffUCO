from pathlib import Path
import pickle
from collections import Counter
import igraph as ig
import jraph
import numpy as np
from DatasetCreator.Gurobi import GurobiSolver
# from DatasetCreator import Gurobi.GurobiSolver
# from Gurobi import GurobiSolver
from tqdm import tqdm
# path = "/home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl"
# def split(path):
#     with open(path, 'rb') as f:W
#         graphs = pickle.load(f)
    
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

def solve_graph(self, H_graph, g):
    """
    Solve the graph instance for the dataset using gurobi if self.gurobi_solve is True, otherwise return None Tuple

    :param H_graph: jraph graph instance
    :param g: igraph graph instance
    :return: (Energy, boundEnergy, solution, runtime, H_graph_compl)
    """
    if self.gurobi_solve:
        H_graph_compl = from_igraph_to_jgraph(g, double_edges=False)
        _, Energy, solution, runtime = GurobiSolver.solveMIS_as_MIP(H_graph, time_limit=self.time_limit)
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

def split(path, dataset_name):
    # path = "/home/chenhaojun/DIffUCO/draft/Data_for_solver_3.pkl"
    # dataset_name = "KS_3"
    with open(path, 'rb') as f:
        graphs = pickle.load(f)
        # graphs.keys()=dict_keys(['unit_vectors', 'overlap_id'])
        n_graph = len(graphs['unit_vectors'])
        save_dataset_dir = Path.cwd() / "Data" / dataset_name
        save_dataset_dir.mkdir(parents=True, exist_ok=True)
        for idx in tqdm(range(n_graph)):
            # edges = Counter(graphs['overlap_id'][idx])
            # g = ig.Graph([(edge[0], edge[1]) for edge in edges])
            # H_graph, density, graph_size = igraph_to_jraph(g)
            # Energy, boundEnergy, solution, runtime, compl_H_graph = solve_graph(H_graph,g)
            save_graph_path = save_dataset_dir / f"{str(idx)}.pickle"
            graph = {}
            graph['unit_vector'] = graphs['unit_vectors'][idx]
            graph['overlap_id'] = graphs['overlap_id'][idx]
            with open(str(save_graph_path), 'wb') as f:
                pickle.dump(graph, f)
                print(f"save graph in {str(save_graph_path)}")

def group_sub_graph(source_dir, target_root_dir, num_folders=8):
    # source_dir = Path("/home/chenhaojun/DIffUCO/Data/KS_3")
    # target_root_dir = Path("/home/chenhaojun/DIffUCO/Data/KS_3_split")
    import shutil
    source_dir = Path(source_dir)
    target_root_dir = Path(target_root_dir)
    files = sorted(source_dir.glob('*.pickle'), key=lambda x: int(x.stem))  # 按照文件名中的数字排序

    num_files = len(files)
    # num_folders = 8
    files_per_folder = num_files // num_folders

    for i in tqdm(range(num_folders)):
        folder_name = f"folder_{i+1}"
        folder_path = target_root_dir / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        
        start_index = i * files_per_folder
        end_index = (i + 1) * files_per_folder if i != num_folders - 1 else num_files
        print(f"start_index={start_index},end_index={end_index}")
        for file in files[start_index:end_index]:
            shutil.move(str(file), str(folder_path))
            print(f"file={file}")

    print("文件移动完成！")

group_sub_graph(source_dir="/home/chenhaojun/DIffUCO/Data/KS_3", target_root_dir="/home/chenhaojun/DIffUCO/Data/KS_3_split", num_folders=8)
import pickle
from collections import Counter
import igraph as ig
import jraph
import numpy as np
import sys
from tqdm import tqdm
from pathlib import Path
import shutil
import argparse
import time
from copy import deepcopy as dp
from tqdm import trange
import numpy as np
import gurobipy as gp
import multiprocessing


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

def solveMIS_as_MIP(H_graph,  time_limit = float("inf"), thread_fraction = 0.5, num_CPUs = None):

    num_nodes = H_graph.nodes.shape[0]
    m = gp.Model("mip1")
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", time_limit)

    if(num_CPUs == None):
        print("Default value of the Threads parameter:", m.Params.Threads)
        m.setParam("Threads", int(thread_fraction*multiprocessing.cpu_count()))
    else:
        m.setParam("Threads", int(num_CPUs))

    edge_list = [(min([s,r]),max([s,r]))for s,r in zip(H_graph.senders, H_graph.receivers)]
    unique_edge_List = set(edge_list)

    var_dict = m.addVars(num_nodes, vtype=gp.GRB.BINARY)
    obj2 = gp.quicksum( -var_dict[int(n)]  for n in range(num_nodes))

    for (s,r) in unique_edge_List:
        xs = var_dict[s]
        xr = var_dict[r]
        m.addConstr(xs + xr <= 1, name="e%d-%d" % (s, r))

    m.setObjective(obj2, gp.GRB.MINIMIZE)
    m.optimize()

    cover = []
    for v in var_dict:
        #print( vertexVars[v].X)
        if var_dict[v].X > 0.5:
            #print ("Vertex'," +str(v)+ 'is in the cover')
            cover.append(v)
    #print("MIS size",len(cover))
    return m, -len(cover), np.array([var_dict[key].x for key in var_dict]), m.Runtime

def solve_graph(H_graph, g, gurobi_solve=True):
    """
    Solve the graph instance for the dataset using gurobi if self.gurobi_solve is True, otherwise return None Tuple

    :param H_graph: jraph graph instance
    :param g: igraph graph instance
    :return: (Energy, boundEnergy, solution, runtime, H_graph_compl)
    """
    if gurobi_solve:
        H_graph_compl = from_igraph_to_jgraph(g, double_edges=False)
        _, Energy, solution, runtime = solveMIS_as_MIP(H_graph)
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

def is_valid_combination(combination, matrix):
    for i in range(len(combination)):
        for j in range(i + 1, len(combination)):
            if matrix[combination[i], combination[j]] > 0.5:
                return False
    return True

def find_combinations(matrix):
    n = matrix.shape[0]
    result = []

    def backtrack(start, current_combination):
        if is_valid_combination(current_combination, matrix):
            result.append(current_combination[:])
        
        for i in range(start, n):
            current_combination.append(i)
            backtrack(i + 1, current_combination)
            current_combination.pop()

    backtrack(0, [])
    results = [k for k in result if len(k)>1]
    return results

def check_overlap(matrix):
    overlapping_pairs = []
    num_balls = matrix.shape[0]
    
    for i in range(num_balls):
        for j in range(i + 1, num_balls):
            if matrix[i, j] > 0.5:
                overlapping_pairs.append((i, j))
    return overlapping_pairs


def cosine_matrix(vectors):
    dim = len(vectors[0])
    nums = len(vectors)
    Matrix = np.zeros((nums, nums))
    
    for i in range(nums):
        for j in range(nums):
            Matrix[i][j] = np.dot(vectors[i], vectors[j].T)

    return Matrix


def generate_unit_vectors(n, num_samples):
    """
    生成 num_samples 个 n 维的单位向量。

    参数:
    n (int): 向量的维度。
    num_samples (int): 生成的单位向量的数量。

    返回:
    numpy.ndarray: 形状为 (num_samples, n) 的单位向量数组。
    """
    # 生成随机向量
    seed = int(time.time())
    #seed = 1
    np.random.seed(seed)
    random_vectors = np.random.randn(num_samples, n)
    
    # 归一化向量以获得单位向量
    norms = np.linalg.norm(random_vectors, axis=1, keepdims=True)
    unit_vectors = random_vectors / norms

    np.random.seed(int(time.time()))
    
    return unit_vectors

# path = "/home/chenhaojun/DIffUCO/draft/Data_for_solver.pkl"
# with open(path, 'rb') as f:
#     graphs = pickle.load(f)
# ipdb.set_trace()
# print(len(graphs)) # 2
# # graphs.keys() = dict_keys(['unit_vectors', 'overlap_id'])
# # len(graphs['overlap_id']) = 100 意思是100张图
# # graphs['overlap_id'][0] 第0张图的邻接表, list存了一堆元组
# # graphs['unit_vectors'][0].shape=(1000, 3)

def get_random_instance(dim, num_samples):
    # Data = {
    #     "unit_vectors":[],
    #     "overlap_id":[]
    # }
    Data = {}
    unit_vectors = generate_unit_vectors(dim, num_samples)
    matrix = cosine_matrix(unit_vectors)
    overlap_id = check_overlap(matrix)
    #print("unit_vectors=",unit_vectors)
    # print("cosin_matrix=",matrix)
    #print("overlap_id=",check_overlap(matrix))
    #print("indenpend set=",combinations)    
    
    Data["unit_vectors"] = unit_vectors
    Data["overlap_id"] = overlap_id

    return Counter(Data['overlap_id']), unit_vectors

parser = argparse.ArgumentParser()
parser.add_argument('--save_graph_dir', default="solution/", type = str)
parser.add_argument('--dim', default=3, type = int, help='generate ndim graphs')
parser.add_argument('--num_samples', default=1000, type = int, help='the number of nodes of generated graphs')
parser.add_argument('--n_graph', default=1, type = int, help='the number of graphs')
args = parser.parse_args()

save_graph_dir = Path(args.save_graph_dir)
save_graph_dir.mkdir(parents=True, exist_ok=True)
n_graph = args.n_graph

for idx in range(n_graph):
    edges, coordinate = get_random_instance(dim=args.dim, num_samples=args.num_samples)
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
    indexed_solution_dict["coordinate"] = coordinate

    save_path = save_graph_dir / f"solution_{idx}.pickle"
    with open(str(save_path), 'wb') as f:
        pickle.dump(indexed_solution_dict, f)
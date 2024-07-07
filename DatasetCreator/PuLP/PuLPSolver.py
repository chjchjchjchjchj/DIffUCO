import numpy as np
import pulp as pl
import multiprocessing

def get_adjacency_list(edges):
    adjacency_list = {}
    for edge in edges:
        if edge[0] not in adjacency_list:
            adjacency_list[edge[0]] = []
        adjacency_list[edge[0]].append(edge[1])

        if edge[1] not in adjacency_list:
            adjacency_list[edge[1]] = []
        adjacency_list[edge[1]].append(edge[0])
    return adjacency_list

def solveMIS_as_MIP(H_graph, time_limit=60*5, thread_fraction=0.5, num_CPUs=None):

    num_nodes = H_graph.nodes.shape[0]
    prob = pl.LpProblem("Maximum_Independent_Set", pl.LpMaximize)

    # Create decision variables
    var_dict = pl.LpVariable.dicts("x", range(num_nodes), 0, 1, pl.LpBinary)

    # Objective function
    prob += pl.lpSum(var_dict[i] for i in range(num_nodes))

    # Constraints
    edge_list = [(min([s,r]), max([s,r])) for s,r in zip(H_graph.senders, H_graph.receivers)]
    unique_edge_list = set(edge_list)

    for (s, r) in unique_edge_list:
        prob += var_dict[s] + var_dict[r] <= 1

    # Solver settings
    solver = pl.getSolver('PULP_CBC_CMD', timeLimit=time_limit)
    
    if num_CPUs is None:
        num_CPUs = int(thread_fraction * multiprocessing.cpu_count())
    
    solver.threads = num_CPUs
    
    # Solve the problem
    prob.solve(solver)

    cover = []
    for v in range(num_nodes):
        if var_dict[v].varValue > 0.5:
            cover.append(v)
    
    return prob, len(cover), np.array([var_dict[key].varValue for key in var_dict]), prob.solutionTime

# Example usage:
# Assuming H_graph is defined with attributes: nodes, senders, receivers
# model, mis_size, solution, runtime = solveMIS_as_MIP(H_graph)

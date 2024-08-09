import igraph as ig
from tqdm import tqdm
import argparse

def generate_d_regular_graphs(num_graphs, num_nodes, degree):
    graphs = []
    
    for _ in tqdm(range(num_graphs), desc="Generating graphs"):
        # Generate a d-regular graph
        while True:
            try:
                graph = ig.Graph.Degree_Sequence([degree] * num_nodes, method="vl")
                graphs.append(graph)
                break
            except ValueError:
                # This can happen if a degree sequence is not graphical, retry
                continue

    return graphs

parser = argparse.ArgumentParser()
parser.add_argument('--licence_path', default="/system/user/sanokows/", type = str, help='licence base path')
parser.add_argument('--num_graphs', default=5000, type = int, help='the number of graphs')
parser.add_argument('--num_nodes', default=5000, type = int, help='the number of nodes of generated graphs')
parser.add_argument('--degrees', default=[4, 6, 8], type = int, help='Degree of each node')

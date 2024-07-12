from .BaseDatasetGenerator import BaseDatasetGenerator
from .RB_graphs import generate_xu_instances
from tqdm import tqdm
import numpy as np
import igraph as ig
import ipdb
from .KS_graphs import generate_ks_instances
import time
from collections import Counter
import pickle

class KSDatasetGenerator(BaseDatasetGenerator):
	"""
	Class for generating datasets with RB Graphs
	"""
	def __init__(self, config):
		super().__init__(config)
		if self.mode != "test":
			self.p_list = [None]
		else:
			if self.diff_ps:
				self.p_list = np.linspace(0.25, 1, 10)
			else:
				self.p_list = [None]

		self.graph_config = self.__init_graph_config(self.dataset_name)
		print(f'\nGenerating RB {self.mode} dataset "{self.dataset_name}" with {self.graph_config[f"n_{self.mode}"]} instances!\n')
		self.load_graph_path = config["datasets_path"]
		self.uniform_generate_data = config['uniform_generate_data']
		self.dim = config['dim']
		self.num_samples = config['num_samples']
		self.thread_fraction = config['thread_fraction']
		self.st_idx = config['st_idx']
		self.ed_idx = config['ed_idx']
		if self.uniform_generate_data:
			print(f"generate {self.dim} dimension {self.num_samples} nodes graphs, thread_fraction is {self.thread_fraction}!!!!!!!!!!!!!")
		if not self.uniform_generate_data:
			with open(self.load_graph_path, 'rb') as f:
				self.loaded_graphs = pickle.load(f)
				print(f"loading datasets from {self.load_graph_path}")

	def __init_graph_config(self, dataset_name):
		"""
		:param dataset_name: dataset name containing the size of the dataset
		:return: graph_config: parameter config needed to generate graph instances
		"""
		if "small" in dataset_name:
			self.size = "small"
			graph_config = {
				"p_low": 0.3, "p_high": 1,
				"n_min": 200, "n_max": 300,
				"n_low": 20, "n_high": 25,
				"k_low": 5, "k_high": 12,
				"n_train": 4000, "n_val": 500, "n_test": 500
			}
		elif "large" in dataset_name:
			self.size = "large"
			graph_config = {
				"p_low": 0.3, "p_high": 1,
				"n_min": 800, "n_max": 1200,
				"n_low": 40, "n_high": 55,
				"k_low": 20, "k_high": 25,
				"n_train": 4000, "n_val": 500, "n_test": 500
			}
		elif "100" in dataset_name:
			self.size = "100"
			graph_config = {
				"p_low": 0.25, "p_high": 1,
				"n_min": 0, "n_max": np.inf,
				"n_low": 9, "n_high": 15,
				"k_low": 8, "k_high": 11,
				"n_train": 3000, "n_val": 500, "n_test": 500
			}
		elif "200" in dataset_name:
			self.size = "200"
			graph_config = {
				"p_low": 0.25, "p_high": 1,
				"n_min": 0, "n_max": np.inf,
				"n_low": 20, "n_high": 25,
				"k_low": 9, "k_high": 10,
				"n_train": 2000, "n_val": 500, "n_test": 500
			}
		elif "huge" in dataset_name:
			self.size = "1000"
			graph_config = {
				"p_low": 0.25, "p_high": 1,
				"n_min": 0, "n_max": np.inf,
				"n_low": 60, "n_high": 70,
				"k_low": 15, "k_high": 20,
				"n_train": 3000, "n_val": 500, "n_test": 500
			}
		elif "giant" in dataset_name:
			self.size = "2000"
			graph_config = {
				"p_low": 0.25, "p_high": 1,
				"n_min": 0, "n_max": np.inf,
				"n_low": 120, "n_high": 140,
				"k_low": 15, "k_high": 20,
				"n_train": 3000, "n_val": 500, "n_test": 500
			}
		elif "dummy" in dataset_name:
			self.size = "dummy"
			graph_config = {
				"p_low": 0.25, "p_high": 1,
				"n_min": 0, "n_max": np.inf,
				"n_low": 9, "n_high": 15,
				"k_low": 8, "k_high": 11,
				"n_train": 300, "n_val": 500, "n_test": 500
			}
		elif "KS_3" in dataset_name:
			self.size = "1000"
			graph_config = {
				"p_low": 0.25, "p_high": 1,
				"n_min": 0, "n_max": np.inf,
				"n_low": 60, "n_high": 70,
				"k_low": 15, "k_high": 20,
				"n_train": 3000, "n_val": 500, "n_test": 500
				# "n_train": 30, "n_val": 5, "n_test": 5
			}
		elif "KS_4" in dataset_name:
			self.size = "1000"
			graph_config = {
				"p_low": 0.25, "p_high": 1,
				"n_min": 0, "n_max": np.inf,
				"n_low": 60, "n_high": 70,
				"k_low": 15, "k_high": 20,
				"n_train": 3000, "n_val": 500, "n_test": 500
				# "n_train": 30, "n_val": 5, "n_test": 5
			}
		elif "KS_5" in dataset_name:
			self.size = "1000"
			graph_config = {
				"p_low": 0.25, "p_high": 1,
				"n_min": 0, "n_max": np.inf,
				"n_low": 60, "n_high": 70,
				"k_low": 15, "k_high": 20,
				"n_train": 3000, "n_val": 500, "n_test": 500
				# "n_train": 30, "n_val": 5, "n_test": 5
			}
		else:
			raise NotImplementedError('Dataset name must contain either "small", "large", "huge", "giant", "100", "200", "KS_3", "KS_4", "KS_5" to infer the number of nodes')
		return graph_config

	# def generate_dataset(self):
	# 	"""
	# 	Generate a RB graph instances for the dataset
	# 	"""
	# 	for p in self.p_list:
	# 		if (self.diff_ps):
	# 			self.dataset_name = f"RB_iid_{self.size}_p_{p}"
	# 			self.graph_config["n_test"] = 100
	# 		else:
	# 			self.dataset_name = f"RB_iid_{self.size}"
	# 		self.generate_graphs(p)

	def generate_dataset(self):
		"""
		Generate a KS graph instances for the dataset
		"""
		for p in self.p_list:
			if (self.diff_ps):
				self.dataset_name = f"KS_iid_{self.size}_p_{p}"
				self.graph_config["n_test"] = 100
			else:
				# self.dataset_name = f"KS_iid_{self.size}"
				# self.dataset_name = f"{self.dataset_name}_{self.size}"
				self.dataset_name = f"{self.dataset_name}_{self.num_samples}"
			self.generate_graphs(p)

	def generate_graphs(self, p):
		solutions = {
			"Energies": [],
			"H_graphs": [],
			"gs_bins": [],
			"graph_sizes": [],
			"densities": [],
			"runtimes": [],
			"upperBoundEnergies": [],
			"compl_H_graphs": [],
			"p": [],
			"coordinate": [],
		}
		# self.ed_idx = self.graph_config[f"n_{self.mode}"]
		# for idx in tqdm(range(self.graph_config[f"n_{self.mode}"])):
		print(f"start at {self.st_idx}, end at {self.ed_idx} ...")
		for idx in tqdm(range(self.st_idx, self.ed_idx)):
		# for idx in tqdm(range(num_graphs)):
			while True:
				if (not self.diff_ps):
					#print("select new p", p)
					p = np.random.uniform(self.graph_config["p_low"], self.graph_config["p_high"])
				else:
					pass
				#print("curr", p)

				min_n, max_n = self.graph_config["n_min"], self.graph_config["n_max"]
				# n = np.random.randint(self.graph_config["n_low"], self.graph_config["n_high"])
				# k = np.random.randint(self.graph_config["k_low"], self.graph_config["k_high"])
				# ipdb.set_trace()
				# edges = generate_xu_instances.get_random_instance(n, k, p)
				if self.mode == "train":
					selected_idx = idx
				elif self.mode == "val":
					selected_idx = self.graph_config["n_train"] + idx
				elif self.mode == "test":
					selected_idx = self.graph_config["n_train"] + self.graph_config["n_val"] + idx
				
				if self.uniform_generate_data:
					edges, coordinate = generate_ks_instances.get_random_instance(dim=self.dim, num_samples=self.num_samples)
				else: edges = Counter(self.loaded_graphs['overlap_id'][selected_idx])
				g = ig.Graph([(edge[0], edge[1]) for edge in edges])
				# isolated_nodes = [v.index for v in g.vs if v.degree() == 0]
				# g.delete_vertices(isolated_nodes)
				num_nodes = g.vcount()
				if min_n <= num_nodes <= max_n:
					break
			H_graph, density, graph_size = self.igraph_to_jraph(g)
			Energy, boundEnergy, solution, runtime, compl_H_graph = self.solve_graph(H_graph,g, thread_fraction=self.thread_fraction)

			if not self.gurobi_solve:
				if self.problem == "MaxCl" or self.problem == "MIS":
					# Energy = -12
					if self.dim == 2:
						Energy = -6
					elif self.dim == 3:
						Energy = -12
					elif self.dim == 4:
						Energy = -24
					elif self.dim == 5:
						Energy = -41
					elif self.dim == 6:
						Energy = -73
					elif self.dim == 7:
						Energy = -127
					elif self.dim == 8:
						Energy = -240
					elif self.dim == 9:
						Energy = -307
					elif self.dim == 10:
						Energy = -511
					else:
						raise NotImplementedError


			solutions["Energies"].append(Energy)
			solutions["H_graphs"].append(H_graph)
			solutions["gs_bins"].append(solution)
			solutions["graph_sizes"].append(graph_size)
			solutions["densities"].append(density)
			solutions["runtimes"].append(runtime)
			solutions["upperBoundEnergies"].append(boundEnergy)
			solutions["compl_H_graphs"].append(compl_H_graph)
			solutions["p"].append(p)
			if not self.uniform_generate_data:
				solutions["coordinate"].append(self.loaded_graphs['unit_vectors'][selected_idx])
			else:
				solutions["coordinate"].append(coordinate)

			indexed_solution_dict = {}
			for key in solutions.keys():
				if len(solutions[key]) > 0:
					# indexed_solution_dict[key] = solutions[key][idx]
					indexed_solution_dict[key] = solutions[key][idx-self.st_idx]
			self.save_instance_solution(indexed_solution_dict, idx)
		self.save_solutions(solutions)

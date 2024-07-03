import os.path
import copy
import numpy as np
from functools import partial
import jax
from jax import lax
import jax.numpy as jnp
import pickle
import optax
import jraph
from tqdm import tqdm
import wandb
from matplotlib import pyplot as plt
from NoiseDistributions import get_Noise_class
from Trainers import get_Trainer_class
from Networks.DiffModel import DiffModel
from jraph_utils import pmap_batch_U_net_graph_dict_and_pad
from utils.lr_schedule import cos_schedule
from EnergyFunctions import get_Energy_class
from MCMC import MCMCSampler

from Data.LoadGraphDataset import SolutionDatasetLoader
from jax.tree_util import tree_flatten
import time
import jraph_utils
from utils import reshape_utils
from utils import dict_count
import os

import warnings

# def my_formatwarning(message, category, filename, lineno, line=None):
#   print(message, category)
#   # lineno is the line number you are looking for
#   print('file:', filename, 'line number:', lineno)
#
# warnings.formatwarning = my_formatwarning
### Switch warnings on or off
import warnings
warn = 'This is a warning'
exception = 'This is an exception'

def main():
    warnings.warn(warn)
    raise RuntimeError(exception)

class TrainMeanField:
	def __init__(self, config, load_wandb_id = None, eval_step_factor = 1, visualize_MIS=False):
		self.load_wandb_id = load_wandb_id
		jax.config.update('jax_disable_jit', not config["jit"])

		self.path_to_models = os.getcwd() + "/Checkpoints"

		self.config = self._init_config(config)
		self.config["eval_step_factor"] = eval_step_factor
		print(self.config)

		self.seed = self.config["seed"]
		self.key = jax.random.PRNGKey(self.seed)

		# if epoch % save_modulo == 0 the params will be saved
		self.save_modulo = 50

		self.dataset_name = self.config["dataset_name"] # RB_iid_small
		self.problem_name = self.config["problem_name"] # MIS

		if(self.problem_name == "TSP"):
			self.pad_k = 1.
		elif("large" in self.dataset_name):
			self.pad_k = 1.05
		else:
			self.pad_k = 1.3

		# if(len(jax.devices()) > 1):
		# 	self.pad_k = 2.

		print("pad_k is", self.pad_k)

		self.epochs = self.config["N_warmup"] + self.config["N_anneal"] + self.config["N_equil"] # 0 + 2000 + 0
		self.config["epochs"] = self.epochs

		self.lr = self.config["lr"]
		self.N_basis_states = self.config["N_basis_states"] # 30
		self.batch_size = self.config["batch_size"] # 30
		self.random_node_features = self.config["random_node_features"] # True
		self.n_random_node_features = self.config["n_random_node_features"] # 5

		self.relaxed = self.config["relaxed"] # True

		self.T_max = self.config["T_max"] # 0.05
		self.T = self.T_max # 0.05
		self.N_warmup = self.config["N_warmup"] # 0
		self.N_anneal = self.config["N_anneal"] # 2
		self.N_equil = self.config["N_equil"] # 0
		self.loss_alpha = self.config["loss_alpha"] # 0.0
		self.MCMC_steps = self.config["MCMC_steps"] # 0

		self.n_diffusion_steps = self.config["n_diffusion_steps"] # 1
		self.mode = self.config["mode"] # diffusion
		self.beta_factor = self.config["beta_factor"] # 0.1

		# Network

		if(self.problem_name == "TSP"):
			self.config["edge_updates"] = True
			if("20" in self.dataset_name):
				self.n_bernoulli_features = 20
			elif("100" in self.dataset_name):
				self.n_bernoulli_features = 100
		else:
			self.n_bernoulli_features = 2

		self.config["n_bernoulli_features"] = self.n_bernoulli_features

		if(self.problem_name == "TSP"):
			self.config["n_features_list_prob"] = [120,120,self.n_bernoulli_features]
		else:
			self.config["n_features_list_prob"] = [120,64,2]

		self.n_features_list_prob = self.config["n_features_list_prob"]
		self.config["n_bernoulli_features"] = self.n_bernoulli_features # 2

		self.n_features_list_nodes = self.config["n_features_list_nodes"] # [64, 64]
		self.n_features_list_edges = self.config["n_features_list_edges"] # [10]
		self.n_features_list_messages = self.config["n_features_list_messages"] # [64, 64]
		self.n_features_list_encode = self.config["n_features_list_encode"] # [30]
		self.n_features_list_decode = self.config["n_features_list_decode"] # [64]
		self.n_message_passes = self.config["n_message_passes"] # [2]
		self.message_passing_weight_tied = self.config["message_passing_weight_tied"] # False
		self.linear_message_passing = self.config["linear_message_passing"] # True

		if("edge_updates" in self.config.keys()):
			self.edge_updates = self.config["edge_updates"] # F
		else:
			self.edge_updates = True

		if("mean_aggr" in self.config.keys()):
			self.mean_aggr = self.config["mean_aggr"]
		else:
			self.mean_aggr = False

		if("noise_potential" in self.config.keys()):
			self.noise_potential = self.config["noise_potential"]
		else:
			self.noise_potential = "annealed_obj"

		if("project_name" in self.config.keys()):
			self.project_name = self.config["project_name"]
		else:
			self.project_name = ""

		if("grad_clip" in self.config.keys()):
			self.grad_clip = self.config["grad_clip"]
		else:
			self.grad_clip = True
			self.config["grad_clip"] = self.grad_clip

		if ("time_conditioning" in self.config.keys()):
			self.time_conditioning = self.config["time_conditioning"]
		else:
			self.time_conditioning = False

		if self.config["wandb"]:
			self.wandb_mode = "online"
		else:
			self.wandb_mode = "disabled"

		config = self.config

		self.wandb_project = f"{self.project_name}{config['mode']}_{config['dataset_name']}_{config['problem_name']}_relaxed_{config['relaxed']}_deeper"
		if config['T_max'] > 0.:
			self.wandb_group = f"{config['seed']}_LMP_T_{config['T_max']}_noise_potential_{config['noise_potential']}_anneal_{config['N_anneal']}_MPasses_{config['n_message_passes']}"
		else:
			self.wandb_group = f"{config['seed']}_LMP_T_{config['T_max']}_anneal_{config['N_anneal']}_MPasses_{config['n_message_passes']}"

		wandb_run = f"lr_{config['lr']}_nh_{config['n_hidden_neurons']}_time_cond_{self.time_conditioning }_n_diff_{config['n_diffusion_steps']}_deeper"

		self.wandb_run_id = wandb.util.generate_id()
		self.wandb_run = f"{self.load_wandb_id}_{self.wandb_run_id}_{wandb_run}"

		self.best_rel_error = float('inf')
		self.best_energy = float("inf")

		self.stop_epochs = self.config["stop_epochs"] # 800
		self.epochs_since_best = 0
		self.visualize_MIS = visualize_MIS
		self.__init_Energy_functions()
		self.__init_noise_distribution_class()
		self.config.pop("vmapped_energy_loss_func")
		self.config.pop("vmapped_energy_func")
		self.__init_dataset()
		self.__init_network()
		self.__init__Trainer()
		self.__init_optimizer_and_params()
		self.__init_functions()
		self.__init_wandb(self.config)
		#self.__init_beta_list()

	def __init__Trainer(self):
		TrainerClass_func = get_Trainer_class(self.config)
		self.TrainerClass = TrainerClass_func(self.config, self.EnergyClass, self.NoiseDistrClass, self.model)

	def __init_Energy_functions(self):
		EnergyClass = get_Energy_class(self.config)
		self.EnergyClass = EnergyClass
		self.relaxed_energy = EnergyClass.calculate_Energy
		self.relaxed_Energy_for_Loss = EnergyClass.calculate_Energy_loss
		self.vmapped_relaxed_energy = jax.vmap(self.relaxed_energy, in_axes=(None, 1, None), out_axes=(1))
		self.vmapped_relaxed_energy_for_Loss = jax.vmap(self.relaxed_Energy_for_Loss, in_axes=(None, 1, None),
														out_axes=(1))
		self.config["vmapped_energy_loss_func"] = self.vmapped_relaxed_energy_for_Loss
		self.config["vmapped_energy_func"] = self.vmapped_relaxed_energy

	def __init_noise_distribution_class(self):
		self.NoiseDistrClass = get_Noise_class(self.config)

	def __init_MCMCSampler(self):
		self.MCMCSamplerClass = MCMCSampler.MCMCSamplerClass(self.model, self.TrainerClass.evaluation_step , self.EnergyClass, self.NoiseDistrClass)


	def _load_last_epoch(self):
		wandb_run_id = self.load_wandb_id
		path_folder = f"{self.path_to_models}/{wandb_run_id}/"
		file_name = f"{wandb_run_id}_last_epoch.pickle"
		with open(path_folder + file_name, "rb") as f:
			loaded_dict = pickle.load(f)
		return loaded_dict

	def _init_config(self, config):
		if(self.load_wandb_id == None):
			return config
		else:
			loaded_dict = self._load_last_epoch()
			loaded_config = loaded_dict["config"]
			return loaded_config

	def __init_network(self):
		"""
		initialize network and optimizer
		"""
		self.graph_mode = self.config["graph_mode"]

		if("graph_norm" in self.config.keys()):
			self.graph_norm = self.config["graph_norm"]
		else:
			self.graph_norm = False

		self.model = DiffModel(n_features_list_prob=self.n_features_list_prob,
								n_features_list_nodes=self.n_features_list_nodes,
								n_features_list_edges=self.n_features_list_edges,
								n_features_list_messages=self.n_features_list_messages,
								n_features_list_encode=self.n_features_list_encode,
								n_features_list_decode=self.n_features_list_decode,
								n_diffusion_steps = self.n_diffusion_steps,
								n_message_passes=self.n_message_passes,
								message_passing_weight_tied=self.message_passing_weight_tied,
								linear_message_passing=self.linear_message_passing,
								edge_updates = self.edge_updates,
								problem_type = self.problem_name,
								n_bernoulli_features = self.n_bernoulli_features,
								mean_aggr = self.mean_aggr,
							   EncoderModel = self.graph_mode, n_random_node_features = self.n_random_node_features,
							   train_mode = self.config["train_mode"],
							   graph_norm = self.graph_norm)


	def __init_optimizer_and_params(self):
		if(self.load_wandb_id == None):
			self.curr_epoch = 0
			self.__init_params()
			self.__init_optimizer(self.lr, self.params)
		else:
			loaded_dict = self._load_last_epoch()

			self.curr_epoch = loaded_dict["epoch"]
			self.params = loaded_dict["params"]

			self.__init_optimizer(self.lr, self.params)
			self.__init_params()

	def __init_optimizer(self, lr, params):
		# self.optimizer = optax.radam(learning_rate=self.curr_lr)
		self.epoch_length = len(self.dataloader_train)*self.TrainerClass.inner_update_steps
		if(self.config["grad_clip"]):
			opt = optax.chain(optax.clip_by_global_norm(1.0), optax.scale_by_radam(),
										 optax.scale_by_schedule(lambda step: -cos_schedule(step, self.epoch_length*(self.N_anneal + self.N_warmup), max_lr=lr, min_lr = lr/10)))
			opt_init, self.opt_update = opt

		else:
			# opt_init, self.opt_update = optax.chain(optax.clip(1.0), optax.scale_by_radam(),
			# 							 optax.scale(-lr))

			# optimizer = optax.adam(learning_rate=lr)
			# self.opt_update = optimizer.update
			# opt_init = optimizer.init

			opt_init, self.opt_update = optax.chain( optax.scale_by_radam(),
										 optax.scale_by_schedule(lambda step: -cos_schedule(step, self.epoch_length*(self.N_anneal + self.N_warmup), max_lr=lr, min_lr = lr/10)))

		if(self.load_wandb_id == None):
			self.opt_state = jax.pmap(opt_init)(params)

		else:
			loaded_dict = self._load_last_epoch()
			self.opt_state = loaded_dict["opt_state"]
			self.opt_state = jax.tree_map(lambda x: x[0], self.opt_state)
			self.opt_state = jax.device_put_replicated(self.opt_state, list(jax.devices()))

		self.TrainerClass.opt_update = self.opt_update

	def __update_lr(self, epoch):
		lr = cos_schedule(epoch, self.N_anneal+ self.N_warmup, max_lr=self.lr, min_lr = self.lr/10)
		return lr

	def __init_functions(self):
		"""
		initialize functions (for jitting or vmapping)
		"""
		pass

	def __init_params(self):
		"""
		initialize network parameters
		"""
		self.key, subkey = jax.random.split(self.key)
		jraph_graph_dict = next(iter(self.dataloader_val))

		if (self.load_wandb_id != None):
			self.params = jax.tree_map(lambda x: x[0], self.params)
		elif(self.graph_mode != "U_net"):

			input_graph_list, energy_graphs = self._prepare_graphs(jraph_graph_dict)

			batched_graph = input_graph_list["graphs"][0]
			X_prev = jnp.ones((batched_graph.nodes.shape[1], 1))
			rand_node_features = jnp.ones((batched_graph.nodes.shape[1], self.n_random_node_features))

			input_graph_list = {"graphs": [jax.tree_util.tree_map(lambda x: x[0], input_graph_list["graphs"][0])]}
			self.params = self.model.init({"params": subkey}, input_graph_list, X_prev, rand_node_features, 0, subkey)

		elif(self.graph_mode == "U_net"):
			reps = 10
			iters = len(self.dataloader_train)*reps
			U_net_graph_dict = jraph_graph_dict["U_net_graph_dict"][0]
			U_net_graph_dict = jax.tree_map(lambda x: jnp.array(x), U_net_graph_dict)
			print(jax.tree_map(lambda x: x.shape, U_net_graph_dict))
			#batched_U_net_graph_dict = batch_U_net_graph_dict(jraph_graph_dict["U_net_graph_dict"])
			#batched_U_net_graph_dict_2 = pmap_batch_U_net_graph_dict_and_pad(jraph_graph_dict["U_net_graph_dict"])
			input_graph_list, energy_graphs = self._prepare_graphs(jraph_graph_dict)

			if(False):
				axis = 1
				batched_U_net_graph_dict = pmap_batch_U_net_graph_dict_and_pad(jraph_graph_dict["U_net_graph_dict"], self.pad_k)
				graph_size_dict = {}
				for graph_types in batched_U_net_graph_dict.keys():
					graph_size_dict[graph_types] = {"edges": [[] for _ in range(iters)], "nodes": [[] for _ in range(iters)]}


				graph_shape_list = []
				for rep in range(reps):
					for j, jraph_graph_dict in enumerate(self.dataloader_train):
						i = len(self.dataloader_train)*rep + j
						batched_U_net_graph_dict = pmap_batch_U_net_graph_dict_and_pad(jraph_graph_dict["U_net_graph_dict"], k=1.2)
						graph_shape = jax.tree_map(lambda x: x.shape, batched_U_net_graph_dict)
						graph_shape_list.append(graph_shape)
						print("len graph shape list", len(dict_count.count_same_dicts(graph_shape_list)), i)

						for graph_types in batched_U_net_graph_dict.keys():

							#print(graph_types, type(batched_U_net_graph_dict[graph_types]))
							if(graph_types != "bottleneck_graph"):
								for layer_graph in batched_U_net_graph_dict[graph_types]:
									#print(layer_graph.nodes.shape, layer_graph.edges.shape, layer_graph.globals.shape)
									graph_size_dict[graph_types]["edges"][i].append(layer_graph.nodes.shape[axis])
									graph_size_dict[graph_types]["nodes"][i].append(layer_graph.edges.shape[axis])
							else:
								layer_graph = batched_U_net_graph_dict[graph_types]
								#print(layer_graph.nodes.shape, layer_graph.edges.shape, layer_graph.globals.shape)
								graph_size_dict[graph_types]["edges"][i].append(layer_graph.nodes.shape[axis])
								graph_size_dict[graph_types]["nodes"][i].append(layer_graph.edges.shape[axis])

				result = dict_count.count_same_dicts(graph_shape_list)
				result_strings = {str(k): v for k, v in result.items()}

				for idx1, el1 in enumerate(result_strings):
					for idx2, el2 in enumerate(result_strings):
						if(idx1 != idx2):
							difference = dict_count.compare_and_replace(el1, el2)
							print(el1)
							print(el2)
							print(el1 == el2)
							print(difference)
							print("\n")

				for idx, key in enumerate(graph_size_dict.keys()):
					plt.figure()
					for list_idx in range(len(graph_size_dict[key]["edges"])):
						plt.subplot(2,1,1)
						plt.title(key + " edges " )
						x_edges = np.arange(0, len(graph_size_dict[key]["edges"][list_idx]))
						y_edges = graph_size_dict[key]["edges"][list_idx]
						y_nodes = graph_size_dict[key]["edges"][list_idx]
						plt.plot(x_edges, y_edges, "-x", label = "edges")
						plt.yscale("log")
						plt.subplot(2,1,2)
						plt.title(key + " nodes " )
						plt.plot(x_edges, y_nodes, "-x", label = "nodes")
						plt.yscale("log")
				plt.show()
				plt.close("all")
				raise ValueError("")


			#node_features = self.n_diffusion_steps + self.n_random_node_features + self.n_bernoulli_features

			X_prev = jnp.ones((U_net_graph_dict["graphs"][0].nodes.shape[0], 1))
			rand_node_features = jnp.ones((U_net_graph_dict["graphs"][0].nodes.shape[0], self.n_random_node_features))
			self.params = self.model.init({"params": subkey}, U_net_graph_dict, X_prev,rand_node_features, 0, subkey)
			# X_prev = jnp.ones(batched_U_net_graph_dict["graphs"][0].nodes.shape[:-1] +(node_features,))
			# self.model.apply(self.params, batched_U_net_graph_dict, X_prev)
		else:
			raise ValueError("")


		num_gpus = jax.local_device_count()
		print("Training is distributed across ", num_gpus, "devices!")
		# if(num_gpus <= 1):
		# 	pass
		# else:
		self.params = jax.device_put_replicated(self.params, list(jax.devices()))

		print("pmapped params")
		print(jax.tree_map(lambda x: x.shape, self.params))

	def _init_and_test_MCMC_sampler(self):
		self.__init_MCMCSampler()
		self.key, subkey = jax.random.split(self.key)
		batched_key = jax.random.split(subkey, num=len(jax.devices()))

		jraph_graph_dict = next(iter(self.dataloader_val))

		input_graph_list, energy_graphs = self._prepare_graphs(jraph_graph_dict)
		T = 1.
		bin_sequence = jnp.ones((energy_graphs.nodes.shape[0], self.n_diffusion_steps + 1, energy_graphs.nodes.shape[1], self.N_basis_states, 1))
		self.MCMCSamplerClass.update_buffer(self.params, input_graph_list, energy_graphs, bin_sequence, batched_key, T)

	def __init_dataset(self):
		self.data_generator = SolutionDatasetLoader(config = self.config, dataset=self.dataset_name,
											   problem=self.problem_name,
											   batch_size=self.batch_size,
											   relaxed=self.relaxed,
											   seed=self.seed)
		self.dataloader_train, self.dataloader_test, self.dataloader_val, (
			self.mean_energy, self.std_energy) = self.data_generator.dataloaders()

	def __init_wandb(self, config):
		"""
		initialize weights and biases

		@param project: project name
		"""
		if(self.config["wandb"]):
			wandb.init(project=self.wandb_project, name=self.wandb_run, group=self.wandb_group, id=self.wandb_run_id,
				   config=config, mode=self.wandb_mode, settings=wandb.Settings(_service_wait=300))


	@partial(jax.jit, static_argnums=(0,))
	def __update_params(self, params, grads, opt_state):
		grad_update, opt_state = self.opt_update(grads, opt_state, params)
		params = optax.apply_updates(params, grad_update)
		return params, opt_state

	@partial(jax.jit, static_argnums=(0,))
	def jittet_tree_mean(self, grad):
		return jax.tree_map(lambda x: jnp.mean(x, axis = 0),grad)


	def __linear_annealing(self, epoch):
		if epoch < self.N_warmup:
			T_curr = self.T_max
		elif epoch >= self.N_warmup and epoch < self.epochs - self.N_equil - 1:
			T_curr = max([self.T_max - self.T_max * (epoch - self.N_warmup) / self.N_anneal, 0])
		else:
			T_curr = 0.

		return T_curr

	def __linear_annealing_reverse(self, epoch):
		if epoch <= self.epochs:
			T_curr = max([self.T_max + epoch / self.N_warmup, 0])
		return T_curr

	def _update_MCMCBuffer_sample(self, graph_batch, energy_graph_batch, bin_sequence, batched_key, T):
		best_MCMC_dict, MCMC_Energy, key = self.MCMCSamplerClass.update_buffer(self.params, graph_batch, energy_graph_batch, bin_sequence,
											batched_key, T, n_steps=self.MCMC_steps)

		return best_MCMC_dict["bin_sequence"]

	def _overwrite_MCMCBuffer_sample(self, bin_sequence, energy_graph, batch_dict):
		dataset_indices = batch_dict["idx"]
		dataset_rand_idxs = batch_dict["rand_idxs"]
		energy_graph = energy_graph._replace(nodes = jnp.swapaxes(bin_sequence, 1,2))
		### TODO this has to be pmap unbatch
		map_func = lambda idx: jraph_utils.unpmap_graph(np.array(energy_graph.n_node), np.array(energy_graph.nodes), idx)

		result = map(map_func, np.arange(0, energy_graph.nodes.shape[0]))
		p_maped_unbatched_X_seq_list = list(result)

		unbatched_X_seq_list = []
		for el in p_maped_unbatched_X_seq_list:
			unbatched_X_seq_list.extend(el)

		update_function = lambda MCMC_seq, idx, rand_idxs: self.data_generator.dataset_train.update_MCMC_buffer(MCMC_seq, idx, rand_idxs)
		mapper = map(update_function, unbatched_X_seq_list, dataset_indices, dataset_rand_idxs)
		list(mapper)

	def _on_epoch_end(self, epoch):
		if (self.MCMC_steps != 0 and epoch != 0):
			self.dataloader_train = self.data_generator.reinint_train_dataloader(int(epoch))
			wandb.log({"MCMC/Energy": np.mean(self.MCMCSamplerClass.MCMC_Energ_list)})
			self.MCMCSamplerClass._reset_MCMC_Energy_list()

	def train_step(self, batch_dict):
		### TODO add code that switches of the buffer
		step1 = time.time()
		graph_batch, energy_graph_batch = self._prepare_graphs(batch_dict)
		step2 = time.time()
		batching_time = step2 - step1

		self.params, self.opt_state, loss, (log_dict, energy_graph_batch, self.key) = self.TrainerClass.train_step(self.params, self.opt_state, graph_batch,
																							  energy_graph_batch, self.T, self.key)

		return loss, (log_dict, energy_graph_batch, batching_time)

	def train(self):
		print("first evaluation...")
		self.save_metrics_dict = {}
		self.save_metrics_dict["eval/energy"] = []
		self.save_metrics_dict["eval/gt_energy"] = []
		self.save_metrics_dict["eval/rel_error"] = []
		self.__save_params_every_epoch(0)
		self.eval(epoch=0)
		print("start training...")
		epoch_range = np.arange(self.curr_epoch, self.epochs)

		#graph_shape_list = []
		for epoch in tqdm(epoch_range, desc="Training"):
			start_train_time = time.time()

			self.T = self.__linear_annealing(epoch)

			### TODO move code that updates MCMC buffer to this palce and update the MCMC buffer for a larger batchsize
			step4 = time.time()
			wandb_log_dict = {}
			epoch_time_dict = {}
			epoch_time_dict["epoch_time/dataloader"] = []
			epoch_time_dict["epoch_time/batching"] = []
			epoch_time_dict["epoch_time/backprob"] = []
			epoch_time_dict["epoch_time/logging"] = []
			for iter, (batch_dict) in enumerate(self.dataloader_train):
				gt_normed_energies = batch_dict["energies"]
				print("batch", iter, "of", len(self.dataloader_train))
				print("batchsize is", len(gt_normed_energies))

				step1 = time.time()
				loss, (log_dict, energy_graph_batch, batching_time) = self.train_step(batch_dict)
				step3 = time.time()

				if("metrics" in log_dict.keys()):
					log_dict_metrics = jax.tree_map(reshape_utils.unravel_dict, log_dict["metrics"])
					batch_log_dict = self.__calculate_reporting(energy_graph_batch,
						log_dict_metrics["energies"], gt_normed_energies, log_dict_metrics["spin_log_probs"], log_dict_metrics["free_energies"])

					### concatenate along device dim
					energy_dict = {f"energies/{key}": log_dict["energies"][key] for key in log_dict["energies"]}
					for key in batch_log_dict.keys():
						if key not in wandb_log_dict:
							wandb_log_dict[key] = []

						wandb_log_dict[key].append(batch_log_dict[key][None, ...])

					for key in energy_dict.keys():
						if key not in wandb_log_dict:
							wandb_log_dict[key] = []

						wandb_log_dict[key].append(energy_dict[key])

				loss_dict = {f"losses/{key}": log_dict["Losses"][key] for key in log_dict["Losses"]}
				for key in loss_dict.keys():
					if key not in wandb_log_dict:
						wandb_log_dict[key] = []
					wandb_log_dict[key].append(loss_dict[key])

				if("time" in log_dict.keys()):
					loss_dict = {f"time/{key}": np.sum(log_dict["time"][key]) for key in log_dict["time"]}
					for key in loss_dict.keys():
						if key not in wandb_log_dict:
							wandb_log_dict[key] = []
						wandb_log_dict[key].append(loss_dict[key])

				epoch_time_dict["epoch_time/dataloader"].append(step1-step4)
				step4 = time.time()
				epoch_time_dict["epoch_time/backprob"].append(step3 - step1)
				epoch_time_dict["epoch_time/batching"].append(batching_time)
				epoch_time_dict["epoch_time/logging"].append(step4 - step3)

			print("backprobagation time", np.sum(epoch_time_dict["epoch_time/backprob"]))
			print("logging time", np.sum(epoch_time_dict["epoch_time/logging"]))
			print("dataloader time", np.sum(epoch_time_dict["epoch_time/dataloader"]))
			print("batching time", np.sum(epoch_time_dict["epoch_time/batching"]))


			self._on_epoch_end(epoch)

			end_train_time = time.time()
			train_time_needed = end_train_time - start_train_time

			new_lr = np.mean(cos_schedule(self.opt_state[1].count, self.epoch_length * (self.N_anneal + self.N_warmup), max_lr=self.lr, min_lr=self.lr / 10))

			train_log_dict = {
				"train/epoch": epoch,
				"schedules/lr": new_lr,
				"schedules/T": self.T,
				"schedules/time": train_time_needed

			}

			wandb_epoch_time_dict = {}
			for key in epoch_time_dict.keys():
				wandb_epoch_time_dict["train/" + key] = np.sum(epoch_time_dict[key])

			for key in wandb_log_dict.keys():
				#print(type(wandb_log_dict[key]))
				try:
					#print(key)
					#print("shape before concatenation", np.array(wandb_log_dict[key]).shape)
					### concatenate along
					res = np.concatenate(np.concatenate(wandb_log_dict[key], axis=1), axis=0)
					#print(res.shape, "concate result")
					train_log_dict["train/" + key] = np.mean(res)
				except:

					#print("shape jsut calc the mean", np.array(wandb_log_dict[key]).shape)
					train_log_dict["train/" + key] = np.mean(wandb_log_dict[key])

			wandb.log(train_log_dict)
			wandb.log(wandb_epoch_time_dict)

			self.eval(epoch=epoch + 1)

			if self.epochs_since_best == self.stop_epochs:
				# early stopping
				break
		wandb.finish()


	def eval(self, epoch, mode = "eval"):

		dataloader = self.dataloader_val

		wandb_log_dict = {}
		save_metrics_at_epoch = {}
		save_metrics_at_epoch["eval/energy"] = []
		save_metrics_at_epoch["eval/gt_energy"] = []
		save_metrics_at_epoch["eval/rel_error"] = []
		for iter, (batch_dict) in enumerate(dataloader):
			gt_normed_energies = batch_dict["energies"]
			print("batchsize is", len(gt_normed_energies))

			graph_batch, energy_graph_batch = self._prepare_graphs(batch_dict)

			self.key, subkey = jax.random.split(self.key)
			batched_key = jax.random.split(subkey, num = len(jax.devices()))

			loss, (log_dict, _) = self.TrainerClass.evaluation_step(self.params, graph_batch, energy_graph_batch, self.T, batched_key, mode = mode)


			log_dict_metrics = jax.tree_map(reshape_utils.unravel_dict, log_dict["metrics"])
			if("Losses" in log_dict.keys()):
				loss_dict = {f"losses/{key}": log_dict["Losses"][key] for key in log_dict["Losses"]}
				for key in loss_dict.keys():
					if key not in wandb_log_dict:
						wandb_log_dict[key] = []

					wandb_log_dict[key].append(loss_dict[key])

			energy_dict = {f"energies/{key}": log_dict["energies"][key] for key in log_dict["energies"]}

			batch_log_dict = self.__calculate_reporting(energy_graph_batch,
				log_dict_metrics["energies"], gt_normed_energies, log_dict_metrics["spin_log_probs"], log_dict_metrics["free_energies"])

			for key in batch_log_dict.keys():
				if key not in wandb_log_dict:
					wandb_log_dict[key] = []

				wandb_log_dict[key].append(batch_log_dict[key][None, ...])

			for key in energy_dict.keys():
				if key not in wandb_log_dict:
					wandb_log_dict[key] = []

				wandb_log_dict[key].append(energy_dict[key])

			save_metrics_at_epoch["eval/energy"].append(batch_log_dict["mean_energy"])
			save_metrics_at_epoch["eval/gt_energy"].append(batch_log_dict["mean_gt_energy"])
			save_metrics_at_epoch["eval/rel_error"].append(batch_log_dict["rel_error"])

		for key in save_metrics_at_epoch.keys():
			self.save_metrics_dict[key].append(np.mean(np.concatenate(save_metrics_at_epoch[key], axis = 0)))


		self.__plot_figures( log_dict)
		eval_log_dict = {
			f"{mode}/epoch": epoch,
			f"{mode}/epochs_since_best": self.epochs_since_best,
			f"{mode}/best_rel_error": self.best_rel_error,
			f"{mode}/best_energy": self.best_energy,
		}

		for key in wandb_log_dict.keys():
			try:
				res = np.concatenate(np.concatenate(wandb_log_dict[key], axis=1), axis=0)
				eval_log_dict[f"{mode}/" + key] = np.mean(res)
			except:

				eval_log_dict[f"{mode}/" + key] = np.mean(wandb_log_dict[key])

		average_energy = eval_log_dict[f"{mode}/mean_energy"]
		mean_rel_energies = eval_log_dict[f"{mode}/rel_error"]

		if mean_rel_energies < self.best_rel_error:
			self.best_rel_error = mean_rel_energies
			self.epochs_since_best = 0
		else:
			self.epochs_since_best += 1

		if average_energy < self.best_energy:
			self.__save_params(best_run=True, eval_dict=eval_log_dict)
			self.best_energy = average_energy
			self.epochs_since_best = 0
		else:
			self.epochs_since_best += 1

		self.__save_params_every_epoch(epoch)

		wandb.log(eval_log_dict)

	def test(self, mode = "test"):

		dataloader = self.dataloader_test

		wandb_log_dict = {}
		save_metrics_at_epoch = {}
		save_metrics_at_epoch["eval/energy"] = []
		save_metrics_at_epoch["eval/gt_energy"] = []
		save_metrics_at_epoch["eval/rel_error"] = []
		save_metrics_at_epoch["eval/MIS_size"] = []

		time_list = []

		for iter, (batch_dict) in enumerate(dataloader):
			gt_normed_energies = batch_dict["energies"]
			print("batchsize is", len(gt_normed_energies))

			graph_batch, energy_graph_batch = self._prepare_graphs(batch_dict)

			self.key, subkey = jax.random.split(self.key)
			batched_key = jax.random.split(subkey, num = len(jax.devices()))

			start_time = time.time()
			loss, (log_dict, _) = self.TrainerClass.evaluation_step(self.params, graph_batch, energy_graph_batch, self.T, batched_key, mode = mode)
			end_time = time.time()

			time_list.append(end_time-start_time)


			log_dict_metrics = jax.tree_map(reshape_utils.unravel_dict, log_dict["metrics"])
			if("Losses" in log_dict.keys()):
				loss_dict = {f"losses/{key}": log_dict["Losses"][key] for key in log_dict["Losses"]}
				for key in loss_dict.keys():
					if key not in wandb_log_dict:
						wandb_log_dict[key] = []

					wandb_log_dict[key].append(loss_dict[key])

			### TODO fix this logging so that batchsize does not have an effect anymore
			energy_dict = {f"energies/{key}": log_dict["energies"][key] for key in log_dict["energies"]}

			batch_log_dict = self.__calculate_reporting(energy_graph_batch,
				log_dict_metrics["energies"], gt_normed_energies, log_dict_metrics["spin_log_probs"], log_dict_metrics["free_energies"])

			batch_CE_log_dict = self.__calculate_reporting(energy_graph_batch,
				log_dict_metrics["energies_CE"], gt_normed_energies, log_dict_metrics["spin_log_probs"], log_dict_metrics["free_energies"], prefix= "CE")

			for key in batch_log_dict.keys():
				if key not in wandb_log_dict:
					wandb_log_dict[key] = []

				wandb_log_dict[key].append(batch_log_dict[key][None, ...])

			for key in batch_CE_log_dict.keys():
				if key not in wandb_log_dict:
					wandb_log_dict[key] = []

				wandb_log_dict[key].append(batch_CE_log_dict[key][None, ...])

			for key in energy_dict.keys():
				if key not in wandb_log_dict:
					wandb_log_dict[key] = []

				wandb_log_dict[key].append(energy_dict[key])
			if self.visualize_MIS:
				from pathlib import Path
				import pickle
				save_path_dir = Path(os.getcwd()) / "Checkpoints"/ self.load_wandb_id / "visualization" / f"{iter}"
				save_path_dir.mkdir(parents=True, exist_ok=True)
				jgraph_save_path = save_path_dir / "jgraph.pkl"
				solution_save_path = save_path_dir / "solution.pkl"
				with open(jgraph_save_path, 'wb') as f:
					pickle.dump(batch_dict['energy_graph'][0], f)
					
				with open(solution_save_path, 'wb') as f:
					pickle.dump(log_dict['X_0'][..., 0, 1], f)
			if iter > 10:
				import sys
				sys.exit()
					

		eval_log_dict = {
			f"{mode}/epochs_since_best": self.epochs_since_best,
			f"{mode}/best_rel_error": self.best_rel_error,
			f"{mode}/best_energy": self.best_energy,
			f"{mode}/overall_time": np.mean(time_list),
		}

		for key in wandb_log_dict.keys():
			try:
				res = np.concatenate(np.concatenate(wandb_log_dict[key], axis=1), axis=0)
				eval_log_dict[f"{mode}/" + key] = np.mean(res)
			except:
				eval_log_dict[f"{mode}/" + key] = np.mean(wandb_log_dict[key])

		return eval_log_dict

	def _prepare_graphs(self, batch_dict):
		if(self.graph_mode != "Transformer"):
			if(self.graph_mode == "U_net"):
				input_graph_dict = pmap_batch_U_net_graph_dict_and_pad(batch_dict["U_net_graph_dict"], k = self.pad_k)
				_, energy_graph = self._pad_graphs(batch_dict["input_graph"], batch_dict["energy_graph"])
				#self.pad_k = 1.3
			elif(self.graph_mode != "U_net"):
				input_graph, energy_graph = self._pad_graphs(batch_dict["input_graph"], batch_dict["energy_graph"])

				if (self.dataset_name == "TSP_random_20"):
					raise ValueError("TSP shuld not be sued with GNNs anymore")
					input_graph = self._add_positional_embeddings(input_graph)

				input_graph_dict = {"graphs": [input_graph]}
		elif(self.graph_mode == "Transformer"):
			input_graph, energy_graph = self._pad_graphs(batch_dict["input_graph"], batch_dict["energy_graph"])

			# X_pos_encoding = self._add_node_encoding(input_graph)
			# input_graph = input_graph._replace(nodes = X_pos_encoding)
			input_graph_dict = {"graphs": [input_graph]}

		return input_graph_dict, energy_graph

	@partial(jax.jit, static_argnums = (0,))
	def _add_positional_embeddings(self, input_graph, dim = 64, L = 1.42):
		edges = input_graph.edges

		i = jnp.arange(0, dim, 2)
		L_prime_sin = L * ((dim - i + 1) / dim)[None, :]
		L_prime_cos = L * ((dim - i) / dim)[None, :]
		cos_edges = jnp.cos(2*jnp.pi*edges/L_prime_cos)
		sin_edges = jnp.sin(2*jnp.pi* edges/L_prime_sin)
		new_edges = jnp.concatenate([edges, cos_edges, sin_edges], axis = -1)

		input_graph = input_graph._replace(edges = new_edges)
		return input_graph

	@partial(jax.jit, static_argnums = (0,))
	def _add_node_encoding(self, input_graph, dim = 64, L = 1.):
		X_pos = input_graph.nodes

		i = jnp.arange(0, dim, 2)
		L_prime_sin = L * ((dim - i + 1) / dim)[None, None, None, :]
		L_prime_cos = L * ((dim - i) / dim)[None, None, None,:]
		cos_X_pos = jnp.cos(2*jnp.pi*X_pos[..., None]/L_prime_cos)
		sin_X_pos = jnp.sin(2*jnp.pi* X_pos[..., None]/L_prime_sin)
		X_pos_embeddings = jnp.concatenate([X_pos[..., None], cos_X_pos, sin_X_pos], axis = -1)
		X_pos_embeddings = jnp.reshape(X_pos_embeddings, X_pos_embeddings.shape[:-2] + (-1,))
		return X_pos_embeddings

	def _pad_graphs(self, input_graph, energy_graph):
		if(self.graph_mode != "Transformer"):
			if (True):
				#input_graph = pad_graph_to_nearest_power_of_k(input_graph, k=self.pad_k)
				#energy_graph = pad_graph_to_nearest_power_of_k(energy_graph, k=self.pad_k)

				input_graph = jraph_utils.pmap_graph_list(input_graph, k = self.pad_k)
				energy_graph = jraph_utils.pmap_graph_list(energy_graph, k = self.pad_k)

			else:
				del energy_graph
				input_graph = jraph_utils.pmap_graph_list(input_graph, k = self.pad_k)
				energy_graph = input_graph
		else:
			input_graph = jraph_utils.pmap_transformer_list(input_graph, k=self.pad_k)
			energy_graph = jraph_utils.pmap_transformer_list(energy_graph, k=self.pad_k)


		return input_graph, energy_graph

	def __plot_figures(self, log_dict, mode = "eval"):
		if("figures" in log_dict.keys()):
			figure_dict = log_dict["figures"]
			plt_dict = {}
			for figure_key in figure_dict:
				fig = plt.figure()
				plt.title(figure_key)

				y_values = np.mean(figure_dict[figure_key]["y_values"],axis = 0)
				x_values = np.arange(0, len(y_values))

				print("figures", y_values.shape)
				print(figure_key)
				print(y_values)

				plt.plot(x_values, y_values, "-x")

				plt_dict[f"{mode}/figures/{figure_key}"] = wandb.Image(fig)
				plt.close("all")

			wandb.log(plt_dict)

	@partial(jax.jit, static_argnums=(0,))
	def calc_mean_prob(self,graphs, spin_log_probs):
		### TODO implement this for more than oe device
		graphs = jax.tree_map(lambda x: jnp.concatenate(x, axis = 0), graphs)
		nodes = graphs.nodes
		n_node = graphs.n_node
		n_graph = jax.tree_util.tree_leaves(n_node)[0].shape[0]
		graph_idx = jnp.arange(n_graph)
		total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
		node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_num_nodes)
		mean_prob_per_graph = jraph.segment_sum(jnp.exp(spin_log_probs), node_graph_idx, n_graph) / n_node[:, None,None]
		return mean_prob_per_graph[:-1]

	def __calculate_reporting(self, graphs, normed_energies, gt_normed_energies, spin_log_probs, normed_free_energies=np.nan, prefix = ""):
		gt_normed_energies = np.array(gt_normed_energies)
		if not np.isnan(normed_free_energies).all():
			mean_normed_free_energy = np.mean(normed_free_energies)
		else:
			mean_normed_free_energy = np.nan

		energies = np.array(normed_energies * self.std_energy + self.mean_energy)
		gt_energies = np.array(gt_normed_energies * self.std_energy + self.mean_energy)
		gt_energies = np.expand_dims(gt_energies, axis=-1)
		gt_energies = gt_energies[:energies.shape[0]]

		min_energies = np.min(energies, axis=1)

		# print("energies", min_energies.shape, energies.shape, gt_energies.shape)
		# print(min_energies - np.max(energies, axis=1))
		rel_error_matrix = np.abs(gt_energies - np.squeeze(energies, axis=-1)) / np.abs(gt_energies)
		best_rel_error_per_graph = np.abs(gt_energies - min_energies) / np.abs(gt_energies)

		mean_normed_energy = normed_energies
		mean_gt_energy = gt_energies
		mean_best_energy = min_energies
		mean_best_rel_error = best_rel_error_per_graph
		#print(rel_error, mean_best_rel_error)


		mean_prob_per_graph = self.calc_mean_prob(graphs, spin_log_probs)

		log_dict = {"mean_energy": energies, "mean_normed_energy": mean_normed_energy,
					"mean_gt_energy": mean_gt_energy, "mean_best_energy":  mean_best_energy, "rel_error": rel_error_matrix,
					"mean_best_rel_error": mean_best_rel_error, "mean_prob": mean_prob_per_graph}

		if self.problem_name == 'MVC':
			APR_per_graph = np.squeeze(energies, axis=-1) / gt_energies
			best_APR_per_graph = min_energies / gt_energies

			log_dict["APR"] = APR_per_graph
			log_dict["best_APR"] = best_APR_per_graph
		if self.problem_name == 'MaxCut':
			energies = np.squeeze(energies, axis = -1)
			num_edges = 2*np.reshape(graphs.n_edge[:,:-1],(graphs.n_edge.shape[0]*(graphs.n_edge.shape[1]-1),1))
			MaxCut_Value = num_edges/4 - energies/2
			best_MaxCut_Value = np.max(num_edges/4 - energies/2, axis = -1)
			gt_MaxCut_Value = num_edges/4 - gt_normed_energies/2

			log_dict["MaxCut_Value"] = MaxCut_Value
			log_dict["best_MaxCut_Value"] = best_MaxCut_Value
		else:
			APR = None
			best_APR = None

		if(prefix != ""):
			new_log_dict = {}
			for key in log_dict:
				new_log_dict[key + f"_{prefix}"] = log_dict[key]
			return new_log_dict

		return log_dict

	def __save_params(self, best_run: bool, eval_dict: dict):
		params_to_save = (self.params, self.config, eval_dict)
		path_folder = f"{self.path_to_models}/{self.wandb_run_id}/"

		if not os.path.exists(path_folder):
			os.makedirs(path_folder)

		if best_run:
			file_name = f"best_{self.wandb_run_id}.pickle"
		else:
			file_name = f"{self.wandb_run_id}_T_{self.T}.pickle"

		with open(os.path.join(path_folder, file_name), 'wb') as f:
			pickle.dump(params_to_save, f)

	def __save_params_every_epoch(self, epoch: int):
		dict_to_save = {"params": self.params,
						"opt_state": self.opt_state,
						"T": self.T,
						"epoch": epoch,
						"config": self.config,
						"logs": self.save_metrics_dict
						}

		path_folder = f"{self.path_to_models}/{self.wandb_run_id}/"

		if not os.path.exists(path_folder):
			os.makedirs(path_folder)

		file_name = f"{self.wandb_run_id}_last_epoch.pickle"

		with open(os.path.join(path_folder, file_name), 'wb') as f:
			pickle.dump(dict_to_save, f)


	def __load_params(self, wandb_id, T, best_run):
		if best_run:
			file_name = f"best_{wandb_id}.pickle"
		else:
			file_name = f"{wandb_id}_T_{round(T, 3)}.pickle"

		with open(f'Checkpoints/{wandb_id}/{file_name}', 'rb') as f:
			params, config = pickle.load(f)
		return params, config

	def __cast_jraph_to(self, j_graph, np_=jnp):
		"""
		cast jraph tuple to np_ (i.e. to np or jnp)

		NOTE: Global features will be ignored; i.e. global features will be set to None!
		"""
		j_graph = jraph.GraphsTuple(nodes=np_.asarray(j_graph.nodes),
									edges=np_.asarray(j_graph.edges),
									receivers=np_.asarray(j_graph.receivers),
									senders=np_.asarray(j_graph.senders),
									n_node=np_.asarray(j_graph.n_node),
									n_edge=np_.asarray(j_graph.n_edge),
									globals=None)
		return j_graph

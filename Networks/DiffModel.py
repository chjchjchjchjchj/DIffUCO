import jax
import numpy as np
import jax.numpy as jnp
import flax
import flax.linen as nn
from functools import partial

from Networks.Modules import get_GNN_model

class DiffModel(nn.Module):
	"""
	Policy Network
	"""
	n_features_list_prob: np.ndarray

	n_features_list_nodes: np.ndarray
	n_features_list_edges: np.ndarray
	n_features_list_messages: np.ndarray

	n_features_list_encode: np.ndarray
	n_features_list_decode: np.ndarray

	n_diffusion_steps: int
	n_message_passes: int
	edge_updates: bool
	problem_type: str
	message_passing_weight_tied: bool = True

	linear_message_passing: bool = True
	n_bernoulli_features: int = 1
	mean_aggr: bool = False
	EncoderModel: str = "normal"
	n_random_node_features: int = 5
	train_mode: str = "REINFORCE"
	graph_norm: bool = False



	def setup(self):
		GNNModel, HeadModel = get_GNN_model(self.EncoderModel, self.train_mode)
		self.encode_process_decode = GNNModel(n_features_list_nodes=self.n_features_list_nodes,
															 n_features_list_edges=self.n_features_list_edges,
															 n_features_list_messages=self.n_features_list_messages,
															 n_features_list_encode=self.n_features_list_encode,
															 n_features_list_decode=self.n_features_list_decode,
															 edge_updates=self.edge_updates,
															 n_message_passes=self.n_message_passes,
															 weight_tied=self.message_passing_weight_tied,
															 linear_message_passing=self.linear_message_passing,
															 mean_aggr = self.mean_aggr,
															 graph_norm = self.graph_norm)

		self.HeadModel = HeadModel(n_features_list_prob=self.n_features_list_prob)

		self.__vmap_get_log_probs = jax.vmap(self.__get_log_prob, in_axes=(0, None, None), out_axes=(0))
		### TODO random node feature key is different during eval and sample, force them to be the same?

	@flax.linen.jit
	def __call__(self, jraph_graph_list, X_prev, rand_node_features, t_idx, key):
		X_prev = self._add_random_nodes_and_time_index(X_prev, rand_node_features, t_idx)
		embeddings = self.encode_process_decode(jraph_graph_list, X_prev)

		bernoulli_embeddings = jnp.repeat(embeddings[:, jnp.newaxis, :], 1, axis = -2)
		embeddings = bernoulli_embeddings

		out_dict = {}
		out_dict = self.HeadModel(jraph_graph_list, embeddings, out_dict)
		out_dict["rand_node_features"] = rand_node_features
		return out_dict, key

	#@partial(flax.linen.jit, static_argnums=0)
	def get_graph_info(self, jraph_graph_list):
		first_graph = jraph_graph_list["graphs"][0]
		nodes = first_graph.nodes
		n_node = first_graph.n_node
		n_graph = jax.tree_util.tree_leaves(n_node)[0].shape[0]
		graph_idx = jnp.arange(n_graph)
		total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
		node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_nodes)
		return node_graph_idx, n_graph, n_node

	@partial(flax.linen.jit, static_argnums=0)
	def reinit_rand_nodes(self, X_t,  key):
		key, subkey = jax.random.split(key)
		rand_nodes = jax.random.uniform(subkey, shape=(X_t.shape[0], self.n_random_node_features))

		return rand_nodes, key

	@partial(flax.linen.jit, static_argnums=(0,))
	def _add_random_nodes_and_time_index(self, X_t, rand_nodes, t_idx):
		time_one_hot = t_idx * jnp.ones_like(X_t[..., -1])
		time_one_hot = jax.nn.one_hot(time_one_hot, num_classes=self.n_diffusion_steps)
		X_one_hot = jax.nn.one_hot(X_t[...,0], num_classes=self.n_bernoulli_features)
		X_input = jnp.concatenate([X_one_hot, time_one_hot, rand_nodes], axis=-1)

		return X_input

	@partial(flax.linen.jit, static_argnums=0)
	def make_one_step(self,params ,jraph_graph_list, X_prev, t_idx, key):
		rand_nodes, key = self.reinit_rand_nodes(X_prev, key)
		out_dict, key = self.apply(params, jraph_graph_list, X_prev, rand_nodes, t_idx, key)

		spin_logits = out_dict["spin_logits"] # spin_logits.shape=(all_nodes, 1, 2)

		node_graph_idx, n_graph, n_node = self.get_graph_info(jraph_graph_list)

		X_next, spin_log_probs, key = self.sample_from_model(spin_logits, key) # (all_nodes, 1), (all_nodes, 1, 2), key
		#print(X_next.shape, X_next, jnp.exp(spin_log_probs))
		graph_log_prob = jax.lax.stop_gradient(jnp.exp((self.__get_log_prob(spin_log_probs[...,0], node_graph_idx, n_graph)/(n_node))[:-1]))
		out_dict["X_next"] = X_next
		out_dict["spin_log_probs"] = spin_log_probs
		out_dict["state_log_probs"] = self.__get_log_prob(spin_log_probs[...,0], node_graph_idx, n_graph)
		out_dict["graph_log_prob"] = graph_log_prob
		return out_dict, key

	@partial(flax.linen.jit, static_argnums=0)
	def unbiased_last_step(self,params ,jraph_graph_list, X_prev, t_idx, key, eps = 0.01):
		rand_nodes, key = self.reinit_rand_nodes(X_prev, key)
		out_dict, key = self.apply(params, jraph_graph_list, rand_nodes, X_prev, t_idx, key)

		spin_logits = out_dict["spin_logits"]
		j_graphs = jraph_graph_list["graphs"][0]
		key, subkey = jax.random.split(key)

		sampled_p = jax.random.uniform(key, shape =  (j_graphs.n_node.shape[0],))

		nodes = j_graphs.nodes
		n_node = j_graphs.n_node
		total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
		graph_sampled_p = jnp.repeat(sampled_p, n_node, axis=0, total_repeat_length=total_nodes)
		graph_sampled_p = graph_sampled_p[:, None]

		X_next_model, spin_log_probs_model, key = self.sample_from_model(spin_logits, key)
		X_next_uniform, one_hot_state, log_p_uniform_density, key = self.sample_prior(j_graphs, spin_logits.shape[1],  key)
		X_next_uniform = X_next_uniform[...,0]
		log_p_uniform = jnp.sum(log_p_uniform_density * one_hot_state, axis=-1)[...,0]

		X_next = jnp.where(graph_sampled_p < eps, X_next_uniform, X_next_model)

		concat_spin_log_probs = jnp.concatenate([spin_log_probs_model[None,...], log_p_uniform[None, ...]], axis = 0)
		weights =  jnp.concatenate([(1-eps)*jnp.ones_like(spin_log_probs_model)[None,...], eps*jnp.ones_like(spin_log_probs_model)[None, ...]], axis = 0)
		spin_log_probs = jax.scipy.special.logsumexp(concat_spin_log_probs, axis = 0, b = weights)

		node_graph_idx, n_graph, n_node = self.get_graph_info(jraph_graph_list)

		graph_log_prob = jax.lax.stop_gradient(jnp.exp((self.__get_log_prob(jnp.sum(spin_log_probs, axis = -1), node_graph_idx, n_graph)/(n_node))[:-1]))
		return X_next, spin_log_probs, spin_logits, graph_log_prob, key

	@partial(flax.linen.jit, static_argnums=0)
	def sample_from_model(self, spin_logits, key):
		key, subkey = jax.random.split(key)
		X_next = jax.random.categorical(key=subkey,
											   logits=spin_logits,
											   axis=-1,
											   shape=spin_logits.shape[:-1])


		one_hot_state = jax.nn.one_hot(X_next, num_classes=self.n_bernoulli_features)
		#X_next = jnp.expand_dims(X_next, axis = -1)
		spin_log_probs = jnp.sum(spin_logits * one_hot_state, axis=-1)

		#print("Diff model model samples", X_next.shape, one_hot_state.shape)
		return X_next, spin_log_probs, key

	@partial(flax.linen.jit, static_argnums=0)
	def calc_log_q(self, params, jraph_graph_list, X_prev, rand_nodes, X_next, t_idx, key):
		out_dict, key = self.apply(params, jraph_graph_list, X_prev, rand_nodes, t_idx, key)

		spin_logits = out_dict["spin_logits"]
		node_graph_idx, n_graph, n_node = self.get_graph_info(jraph_graph_list)

		one_hot_state = jax.nn.one_hot(X_next, num_classes=self.n_bernoulli_features)
		#X_next = jnp.expand_dims(X_next, axis = -1)
		spin_log_probs = jnp.sum(spin_logits * one_hot_state, axis=-1)
		#print(X_next.shape, X_next, jnp.exp(spin_log_probs))
		X_next_log_prob = self.__get_log_prob(spin_log_probs[...,0], node_graph_idx, n_graph)

		# graph_log_prob = jax.lax.stop_gradient(jnp.exp((self.__get_log_prob(spin_log_probs[...,0], node_graph_idx, n_graph)/(n_node[:,None]*self.n_bernoulli_features))[:-1]))
		# print("average prob T:0", jnp.mean(graph_log_prob))
		out_dict["state_log_probs"] = X_next_log_prob
		out_dict["spin_log_probs"] = spin_log_probs
		return out_dict, key

	@partial(flax.linen.jit, static_argnums=0)
	def calc_log_q_T(self, j_graph, X_T):
		'''

		:param j_graph:
		:param X_T: shape =  (batched_graph_nodes, n_states, 1)
		:return:
		'''

		shape = X_T.shape[0:-1]
		log_p_uniform = self._get_prior( shape)

		one_hot_state = jax.nn.one_hot(X_T[...,-1], num_classes=self.n_bernoulli_features)
		log_p_X_T_per_node = jnp.sum(log_p_uniform * one_hot_state, axis=-1)

		nodes = j_graph.nodes
		n_node = j_graph.n_node
		n_graph = j_graph.n_node.shape[0]
		graph_idx = jnp.arange(n_graph)
		total_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0]
		node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_nodes)

		log_p_X_T = self.__get_log_prob(log_p_X_T_per_node, node_graph_idx, n_graph)

		# graph_log_prob = jax.lax.stop_gradient(jnp.exp((self.__get_log_prob(log_p_X_T_per_node, node_graph_idx, n_graph)/(n_node[:,None]*self.n_bernoulli_features))[:-1]))
		# print("average prob 0", jnp.mean(graph_log_prob))
		return log_p_X_T

	@partial(flax.linen.jit, static_argnums=(0,2))
	def sample_prior(self, j_graph, N_basis_states, key):
		nodes = j_graph.nodes
		shape = (nodes.shape[0], N_basis_states, 1)

		key, subkey = jax.random.split(key)
		log_p_uniform = self._get_prior(shape) # 形状为 shape 的 log (1 / 伯努利特征数量)

		X_prev = jax.random.categorical(key=subkey,
										logits=log_p_uniform,
										axis=-1,
										shape=log_p_uniform.shape[:-1]) # (all_nodes, basis_states, 1)

		one_hot_state = jax.nn.one_hot(X_prev, num_classes=self.n_bernoulli_features) # (all_nodes, basis_states, 1, 2)
		return X_prev, one_hot_state, log_p_uniform, key

	@partial(flax.linen.jit, static_argnums=(0,2))
	def sample_prior_w_probs(self, j_graph, N_basis_states, key):
		X_T, one_hot_state, log_p_uniform, key = self.sample_prior(j_graph, N_basis_states, key)
		log_p_X_T = self.calc_log_q_T(j_graph, X_T)
		return X_T, log_p_X_T, one_hot_state, log_p_uniform, key

	@partial(flax.linen.jit, static_argnums=(0,1))
	def _get_prior(self, shape):
		log_p_uniform = jnp.log(1./self.n_bernoulli_features * jnp.ones(shape +  (self.n_bernoulli_features, )))
		return log_p_uniform

	#@partial(flax.linen.jit, static_argnums=(0,-1))
	def __get_log_prob(self, spin_log_probs, node_graph_idx, n_graph):
		log_probs = self.__global_graph_aggr(spin_log_probs, node_graph_idx, n_graph)
		return log_probs

	#@partial(flax.linen.jit, static_argnums=(0,-1))
	def __global_graph_aggr(self, feature, node_graph_idx, n_graph):
		aggr_feature = jax.ops.segment_sum(feature, node_graph_idx, n_graph)
		return aggr_feature



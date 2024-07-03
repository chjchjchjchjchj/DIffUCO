from abc import ABC, abstractmethod
from functools import partial
import jax
import optax
import jax.numpy as jnp

class Base(ABC):
    def __init__(self, config, EnergyClass, NoiseClass, model):
        ### TODO implement learning rate schedule correctly
        self.config = config
        self.n_random_node_features = self.config["n_random_node_features"]
        self.opt_update = None
        self.EnergyClass = EnergyClass
        self.NoiseDistrClass = NoiseClass
        self.model = model
        self.eval_step_factor = self.config["eval_step_factor"]
        print("EVAL STEP FACTOR is", self.eval_step_factor)

        self.vmapped_make_one_step = jax.vmap(self.model.make_one_step, in_axes=(None, None, 1, None, 0),
                                              out_axes=(1, 0))
        self.NoiseDistrClass = NoiseClass
        self.Noise_func = self.NoiseDistrClass.calc_noise_loss
        self.beta_arr = self.NoiseDistrClass.beta_arr
        self.calc_loss = self.NoiseDistrClass.combine_losses

        self.loss_grad = jax.jit(jax.value_and_grad(self.get_loss, has_aux=True))
        self.pmap_sample = jax.pmap(self.sample, in_axes=(0, 0, 0, None, 0))
        self.pmap_update = jax.pmap(self.__update_params, in_axes=(0, 0, 0))
        self.pmap_loss_backward = jax.pmap(self.loss_backward, in_axes=(0, 0, 0, 0, None, 0), axis_name="device")

        self.relaxed_energy = EnergyClass.calculate_Energy
        self.vmapped_relaxed_energy = jax.vmap(self.relaxed_energy, in_axes=(None, 1, None), out_axes=(1,1,1))

        self.energy_CE = EnergyClass.calculate_Energy_CE
        self.vmapped_energy_CE = jax.vmap(self.energy_CE, in_axes=(None,1, None), out_axes=(1))

        self.EnergyClass = EnergyClass
        self.calculate_Energy_CE_p_values = EnergyClass.calculate_Energy_CE_p_values
        self.vmapped_calculate_Energy_CE_p_values = jax.vmap(self.calculate_Energy_CE_p_values, in_axes=(None,1), out_axes=(1))

        self.energy_feasible = EnergyClass.calculate_Energy_feasible
        self.vmapped_energy_feasible = jax.vmap(self.energy_feasible, in_axes=(None, 1), out_axes=(1,0,1))

        self.relaxed_Energy_for_Loss = EnergyClass.calculate_Energy_loss
        self.vmapped_relaxed_energy_for_Loss = jax.vmap(self.relaxed_Energy_for_Loss, in_axes=(None, 1, None),
                                                        out_axes=(1))

        self.pmap_apply_CE_on_p = jax.pmap(self.apply_CE_on_p, in_axes=(0, 0))

    @abstractmethod
    def get_loss(self):
        pass

    @abstractmethod
    def sample(self):
        pass

    def _apply_CE(self):
        pass

    def train_step(self, params, opt_state, graphs, energy_graph_batch, T, key):

        key, subkey = jax.random.split(key)
        batched_key = jax.random.split(subkey, num=len(jax.devices()))

        (loss, (log_dict, _)), params, opt_state = self.pmap_loss_backward_step(params, opt_state, graphs, energy_graph_batch, T, batched_key)
        return params, opt_state, loss, (log_dict, energy_graph_batch, key)

    def evaluation_step(self, params, graph_batch, energy_graph_batch ,T, batched_key, mode = "eval"):
        loss, (log_dict, _) = self.pmap_sample(params, graph_batch, energy_graph_batch, T, batched_key)
        ### TODO add apply CE here

        if(mode == "test"):
            print("testing eval step factor is", self.eval_step_factor)
            if(self.config["problem_name"] != "TSP"):
                p_0 = jnp.exp(log_dict["log_p_0"][...,1])
                X_0_CE, energies_CE, Hb_per_node = self.pmap_apply_CE_on_p(energy_graph_batch, p_0)
                log_dict["metrics"]["X_0_CE"], log_dict["metrics"]["energies_CE"] = X_0_CE, energies_CE[:,:-1]

                # print(jax.tree_map(lambda x: x.shape, energy_graph_batch))
                # print("num_violations")
                # print(jnp.sum((Hb_per_node != 0) * 1.))
                #
                # next_Energy = log_dict["metrics"]["energies_CE"]
                # prev_Energy = log_dict["metrics"]["energies"]
                # print("average ps", jnp.mean(p_0[:, :-1]), prev_Energy.shape, energy_graph_batch.edges.shape, graph_batch["graphs"][0].edges.shape)
                # print("prev Energy", prev_Energy[0,0])
                # print("next energies", next_Energy[0,0])
                # print(jnp.sum((X_0_CE[0,0:energy_graph_batch.n_node[0,0]] != 0 ) * (X_0_CE[0,0:energy_graph_batch.n_node[0,0]] != 1 )*1.))
                #
                # energy_graph_batch_copy = jax.tree_map(lambda x: x[0], energy_graph_batch)
                # node_graph_idx, n_graph, total_num_nodes = self._compute_aggr_utils(energy_graph_batch_copy)
                # Energy, _, _ = self.vmapped_relaxed_energy(energy_graph_batch_copy, X_0_CE[0], node_graph_idx)
                # print("next next energies", Energy[0], Energy.shape, X_0_CE.shape, p_0.shape)
                #
                # Energy, _, _ = self.vmapped_relaxed_energy(energy_graph_batch_copy, X_0_CE[0], node_graph_idx)
                #
                # #self.EnergyClass.calculate_Energy_CE_p_values_debug(energy_graph_batch_copy, p_0[0,:,0])
                #
                # raise ValueError("")
            else:
                log_dict["metrics"]["X_0_CE"] = log_dict["metrics"]["X_0"]
                log_dict["metrics"]["energies_CE"] =  log_dict["metrics"]["energies"]



        return loss, (log_dict, _)

    @partial(jax.jit, static_argnums=(0,))
    def apply_CE_on_p(self, energy_graph_batch, p_0):
        X_0_CE, energies_CE, Hb_per_node = self.vmapped_calculate_Energy_CE_p_values(energy_graph_batch, p_0)

        return X_0_CE, energies_CE, Hb_per_node

    # @abstractmethod
    # def sample(self):
    #     pass

    @partial(jax.jit, static_argnums=(0,))
    def __update_params(self, params, grads, opt_state):
        grad_update, opt_state = self.opt_update(grads, opt_state, params)
        params = optax.apply_updates(params, grad_update)
        return params, opt_state

    def pmap_loss_backward_step(self, params, opt_state, graphs, energy_graph_batch, T, key):
        (loss, (log_dict, key)), params, opt_state = self.pmap_loss_backward(params, opt_state, graphs,
                                                                                   energy_graph_batch, T,
                                                                                   key)

        return (loss, (log_dict, key)), params, opt_state

    @partial(jax.jit, static_argnums=(0,))
    def loss_backward(self, params, opt_state, graphs, energy_graph_batch, T, key):
        (loss, (log_dict, key)), grad = self.loss_grad(params, graphs, energy_graph_batch, T, key)
        grad = jax.lax.pmean(grad, axis_name='device')
        params, opt_state = self.__update_params(params, grad, opt_state)
        return (loss, (log_dict, key)), params, opt_state

    @partial(jax.jit, static_argnums=(0,))
    def _compute_aggr_utils(self, jraph_graph):
        """
        比如一共有三个图
        graph_idx = [0, 1, 2]
        n_node = [3, 5, 2]
        total_num_nodes = [10]
        node_graph_idx = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2]
        """
        nodes = jraph_graph.nodes # 所有图中节点的总和
        n_node = jraph_graph.n_node # 每个子图中节点的个数
        n_graph = jax.tree_util.tree_leaves(n_node)[0].shape[0] # 图的个数
        graph_idx = jnp.arange(n_graph) # 每个图的索引
        total_num_nodes = jax.tree_util.tree_leaves(nodes)[0].shape[0] 
        node_graph_idx = jnp.repeat(graph_idx, n_node, axis=0, total_repeat_length=total_num_nodes)
        return node_graph_idx, n_graph, total_num_nodes


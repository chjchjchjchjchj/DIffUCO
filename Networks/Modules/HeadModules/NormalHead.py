import jax
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from Networks.Modules.MLPModules.MLPs import ProbMLP
from functools import partial
import flax
class NormalHeadModule(nn.Module):
    """
    Multilayer Perceptron with ReLU activation function in the last layer

    @param num_features_list: list of the number of features in the layers (number of nodes); Example: [32, 32, 2] -> two hidden layers with 32 nodes and an output layer with 2 nodes
    """
    n_features_list_prob: np.ndarray

    def setup(self):
        self.probMLP = ProbMLP(n_features_list=self.n_features_list_prob)

    @partial(flax.linen.jit, static_argnums=0)
    def __call__(self, jraph_graph_list, x, out_dict) -> jnp.ndarray:
        """
        forward pass though MLP
        @param x: input data as jax numpy array
        """
        spin_logits = self.probMLP(x)
        out_dict["spin_logits"] = spin_logits

        return out_dict
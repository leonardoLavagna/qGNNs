#------------------------------------------------------------------------------
# shallow_kipf_welling_gnn
#
# This module implements a minimal classical graph neural network layer based
# on the propagation rule proposed by Kipf and Welling [1]. Functions and classes included:
# - KipfWellingLayer:
#   Single graph convolution layer using the normalized adjacency
#   A_hat = D_tilde^{-1/2} (A + I) D_tilde^{-1/2}.
# - relu(x):
#   Elementwise ReLU activation.
# - identity(x):
#   Identity activation.
# - build_identity_weight(in_features):
#   Creates an identity weight matrix.
# - build_random_weight(in_features, out_features, seed):
#   Creates a random weight matrix.
#
# The module is designed to work with GraphInput objects defined in
# utilities/graphs.py and provides a clean classical baseline for later
# comparison with the quantum spectral and Chebyshev-based approaches.
#
# References:
# [1] https://arxiv.org/abs/1609.02907
#
# © Leonardo Lavagna 2026
# leonardo.lavagna@uniroma1.it
#------------------------------------------------------------------------------


from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
from utils.graphs import GraphInput


Activation = Callable[[np.ndarray], np.ndarray]


def relu(x: np.ndarray) -> np.ndarray:
    """
    Apply ReLU activation.

    Args:
        x: Input array.

    Returns:
        Array with negative values replaced by zero.
    """
    return np.maximum(x, 0.0)


def identity(x: np.ndarray) -> np.ndarray:
    """
    Apply identity activation.

    Args:
        x: Input array.

    Returns:
        The input array unchanged.
    """
    return x


def build_identity_weight(in_features: int) -> np.ndarray:
    """
    Build an identity weight matrix.

    Args:
        in_features: Number of input features.

    Returns:
        Identity matrix of shape (in_features, in_features).
    """
    return np.eye(in_features, dtype=float)


def build_random_weight(in_features: int, out_features: int, seed: Optional[int] = None,
                        scale: float = 0.1,) -> np.ndarray:
    """
    Build a random Gaussian weight matrix.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        seed: Optional random seed.
        scale: Standard deviation of the Gaussian.

    Returns:
        Weight matrix of shape (in_features, out_features).
    """
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=scale, size=(in_features, out_features))


@dataclass
class KipfWellingLayer:
    """
    Single Kipf-Welling graph convolution layer.

    The layer applies:
        H_out = activation(A_hat @ H_in @ W + b)

    where:
        - A_hat is the normalized adjacency with self-loops
        - H_in are the node features
        - W is the trainable weight matrix
        - b is an optional bias
    """

    weight: np.ndarray
    bias: Optional[np.ndarray] = None
    activation: Activation = identity

    def forward(self, graph_input: GraphInput, features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply the layer to a graph.

        Args:
            graph_input: GraphInput object.
            features: Optional feature matrix of shape (N, F_in).
                If None, graph_input.node_signals is used.

        Returns:
            Output feature matrix of shape (N, F_out).
        """
        x = self._resolve_features(graph_input, features)
        a_hat = graph_input.kipf_welling_adjacency()
        z = a_hat @ x @ self.weight
        if self.bias is not None:
            z = z + self.bias

        return self.activation(z)

    def _resolve_features(self, graph_input: GraphInput, features: Optional[np.ndarray],) -> np.ndarray:
        """
        Resolve the feature matrix used by the layer.

        Args:
            graph_input: GraphInput object.
            features: Optional external features.

        Returns:
            Feature matrix of shape (N, F).
        """
        if features is None:
            if graph_input.node_signals is None:
                raise ValueError(
                    "No input features provided. "
                    "Pass 'features' explicitly or set graph_input.node_signals.")
            x = graph_input.node_signals
        else:
            x = np.asarray(features, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.shape[0] != graph_input.num_nodes:
            raise ValueError(
                "Feature matrix row count must match the number of graph nodes.")
        if x.shape[1] != self.weight.shape[0]:
            raise ValueError(
                "Feature dimension and weight input dimension do not match.")
        
        return x


def one_step_kipf_welling(graph_input: GraphInput, weight: np.ndarray,
                          bias: Optional[np.ndarray] = None, activation: Activation = identity,
                          features: Optional[np.ndarray] = None,) -> np.ndarray:
    """
    Apply one Kipf-Welling graph convolution step.

    Args:
        graph_input: GraphInput object.
        weight: Weight matrix of shape (F_in, F_out).
        bias: Optional bias of shape (F_out,) or (1, F_out).
        activation: Activation function.
        features: Optional feature matrix. If None, node_signals are used.

    Returns:
        Output feature matrix.
    """
    layer = KipfWellingLayer(weight=weight, bias=bias, activation=activation)
    return layer.forward(graph_input=graph_input, features=features)


def two_layer_kipf_welling(graph_input: GraphInput, weight_1: np.ndarray, weight_2: np.ndarray,
                           bias_1: Optional[np.ndarray] = None, bias_2: Optional[np.ndarray] = None,
                           hidden_activation: Activation = relu, output_activation: Activation = identity,
                           features: Optional[np.ndarray] = None,) -> np.ndarray:
    """
    Apply a simple two-layer Kipf-Welling network.

    Args:
        graph_input: GraphInput object.
        weight_1: First-layer weight matrix.
        weight_2: Second-layer weight matrix.
        bias_1: Optional first-layer bias.
        bias_2: Optional second-layer bias.
        hidden_activation: Activation after first layer.
        output_activation: Activation after second layer.
        features: Optional input features.

    Returns:
        Output feature matrix after two graph convolution layers.
    """
    hidden = one_step_kipf_welling(graph_input=graph_input, weight=weight_1, bias=bias_1,
                                   activation=hidden_activation, features=features,)

    output = one_step_kipf_welling(graph_input=graph_input, weight=weight_2, bias=bias_2,
                                   activation=output_activation, features=hidden,)

    return output

# -----------------------------------------------------------------------------
# graphs.py
#
# This module provides unified utilities and data structures for representing
# graphs together with node and edge signals in a consistent, validated form.
# It is the central graph-operator module of the project and should be the
# single source of truth for all graph-derived matrices used downstream by
# both classical and quantum GNN components. Main responsibilities:
# - validate graph, node order, node signals, and edge signals
# - build adjacency and degree matrices
# - build combinatorial and normalized Laplacians
# - build Kipf-Welling normalized adjacency with self-loops
# - build the associated Laplacian-like operator L_tilde = I - A_hat
# - provide spectral rescaling utilities for symmetric/Hermitian operators
# - provide helper constructors for common default graphs
#
# Classes:
# - GraphInput:
#   Container for an undirected NetworkX graph together with optional node
#   and edge features. It exposes validated graph metadata and methods to
#   compute graph-derived operators used across the pipeline.
#
# Functions:
# - _as_2d_signal_array(...):
#   Convert scalar/vector node signals into a validated 2D NumPy array.
# - _normalize_edge(...):
#   Normalize an undirected edge key into canonical ordered form.
# - build_graph_input(...):
#   Create a GraphInput instance from a NetworkX graph and optional signals.
# - build_from_adjacency(...):
#   Create a GraphInput instance starting from an adjacency matrix.
# - default_two_node(...):
#   Build the default two-node graph as a GraphInput instance.
# - default_path(...):
#   Build a path graph as a GraphInput instance.
# - default_cycle(...):
#   Build a cycle graph as a GraphInput instance.
# - default_complete(...):
#   Build a complete graph as a GraphInput instance.
# - default_star(...):
#   Build a star graph as a GraphInput instance.
#
# This module is designed to support a modular pipeline in which graph
# structure is defined once and then reused consistently by classical
# Kipf-Welling layers, first-order spectral filters, Chebyshev filters,
# exponential quantum filters, and future graph-dependent operators.
#
# © Leonardo Lavagna 2026
# leonardo.lavagna@uniroma1.it
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np


Number = Union[int, float, complex]
EdgeKey = Tuple[int, int]


def _as_2d_signal_array(
    values: Sequence[Union[Number, Sequence[Number]]],
    expected_length: int,
    name: str,
) -> np.ndarray:
    """
    Convert a list of scalar/vector signals into a 2D array.

    Args:
        values: List of scalar or 1D sequences.
        expected_length: Number of nodes expected.
        name: Name used for error messages.

    Returns:
        A NumPy array of shape (N, F).
    """
    if len(values) != expected_length:
        raise ValueError(f"{name} must have length {expected_length}.")

    processed = []
    for i, value in enumerate(values):
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        elif arr.ndim != 1:
            raise ValueError(f"{name}[{i}] must be scalar or 1D.")
        processed.append(arr)

    dims = {arr.shape[0] for arr in processed}
    if len(dims) != 1:
        raise ValueError(f"{name} must have consistent feature size.")

    return np.vstack(processed)


def _normalize_edge(u: int, v: int) -> EdgeKey:
    """
    Normalize an undirected edge key.

    Args:
        u: First node.
        v: Second node.

    Returns:
        Ordered edge tuple (min(u, v), max(u, v)).
    """
    return (u, v) if u <= v else (v, u)


@dataclass
class GraphInput:
    """
    Container for a graph together with optional node and edge signals.

    Parameters:
        graph: NetworkX undirected graph.
        node_order: Ordered list of graph nodes.
        node_signals: Optional array of shape (N, F_node).
        edge_signals: Optional dict mapping (u, v) -> edge feature vector.
    """

    graph: nx.Graph = field(default_factory=lambda: nx.path_graph(2))
    node_order: List[int] = field(
        default_factory=lambda: sorted(nx.path_graph(2).nodes())
    )
    node_signals: Optional[np.ndarray] = None
    edge_signals: Dict[EdgeKey, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Run validation checks after initialization."""
        self._validate_graph()
        self._validate_node_order()
        self._validate_node_signals()
        self._validate_edge_signals()

    def _validate_graph(self) -> None:
        """Ensure the graph is undirected and non-empty."""
        if self.graph.is_directed():
            raise ValueError("Only undirected graphs are supported.")
        if self.graph.number_of_nodes() == 0:
            raise ValueError("Graph cannot be empty.")

    def _validate_node_order(self) -> None:
        """Ensure node_order matches the graph node set exactly."""
        if set(self.node_order) != set(self.graph.nodes()):
            raise ValueError("node_order must match graph nodes exactly.")

    def _validate_node_signals(self) -> None:
        """Validate node signal array shape."""
        if self.node_signals is None:
            return

        arr = np.asarray(self.node_signals, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        if arr.ndim != 2:
            raise ValueError("node_signals must be a 1D or 2D array.")

        if arr.shape[0] != self.num_nodes:
            raise ValueError("node_signals must match the number of nodes.")

        self.node_signals = arr

    def _validate_edge_signals(self) -> None:
        """Validate edge signal keys and feature dimensions."""
        validated: Dict[EdgeKey, np.ndarray] = {}
        graph_edges = {_normalize_edge(int(u), int(v)) for u, v in self.graph.edges()}

        feature_dim = None
        for (u, v), value in self.edge_signals.items():
            key = _normalize_edge(u, v)
            if key not in graph_edges:
                raise ValueError(f"Edge {key} is not present in the graph.")

            arr = np.asarray(value, dtype=float)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            elif arr.ndim != 1:
                raise ValueError("Each edge signal must be scalar or 1D.")

            if feature_dim is None:
                feature_dim = arr.shape[0]
            elif arr.shape[0] != feature_dim:
                raise ValueError("All edge signals must have the same dimension.")

            validated[key] = arr

        self.edge_signals = validated

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes."""
        return len(self.node_order)

    def adjacency_matrix(self) -> np.ndarray:
        """
        Compute the adjacency matrix.

        Returns:
            Adjacency matrix of shape (N, N).
        """
        return nx.to_numpy_array(self.graph, nodelist=self.node_order, dtype=float)

    def degree_matrix(self) -> np.ndarray:
        """
        Compute the degree matrix.

        Returns:
            Degree matrix of shape (N, N).
        """
        adjacency = self.adjacency_matrix()
        degrees = np.sum(adjacency, axis=1)
        return np.diag(degrees)

    def laplacian_matrix(self) -> np.ndarray:
        """
        Compute the combinatorial Laplacian L = D - A.

        Returns:
            Laplacian matrix of shape (N, N).
        """
        return self.degree_matrix() - self.adjacency_matrix()

    def normalized_laplacian_matrix(self) -> np.ndarray:
        """
        Compute the symmetric normalized Laplacian.

        Returns:
            Normalized Laplacian matrix of shape (N, N).

        Raises:
            ValueError: If the graph contains isolated nodes.
        """
        adjacency = self.adjacency_matrix()
        degrees = np.sum(adjacency, axis=1)

        if np.any(degrees == 0):
            raise ValueError("Normalized Laplacian is undefined for isolated nodes.")

        degree_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
        return np.eye(self.num_nodes, dtype=float) - degree_inv_sqrt @ adjacency @ degree_inv_sqrt

    def kipf_welling_adjacency(self) -> np.ndarray:
        """
        Compute the Kipf-Welling normalized adjacency with self-loops:

            A_hat = D_tilde^(-1/2) (A + I) D_tilde^(-1/2)

        Returns:
            Matrix A_hat of shape (N, N).
        """
        adjacency = self.adjacency_matrix()
        adjacency_tilde = adjacency + np.eye(self.num_nodes, dtype=float)
        degrees_tilde = np.sum(adjacency_tilde, axis=1)
        degree_inv_sqrt = np.diag(1.0 / np.sqrt(degrees_tilde))
        return degree_inv_sqrt @ adjacency_tilde @ degree_inv_sqrt

    def kipf_welling_laplacian(self) -> np.ndarray:
        """
        Compute the Laplacian-like operator associated with Kipf-Welling
        normalized adjacency:

            L_tilde = I - A_hat

        Returns:
            Matrix L_tilde of shape (N, N).
        """
        a_hat = self.kipf_welling_adjacency()
        return np.eye(self.num_nodes, dtype=float) - a_hat

    def rescaled_kipf_welling_laplacian(self) -> np.ndarray:
        """
        Compute a spectrally rescaled version of L_tilde = I - A_hat, mapping
        its spectrum approximately into [-1, 1].

        Returns:
            Rescaled matrix of shape (N, N).
        """
        laplacian = self.kipf_welling_laplacian()
        return self.rescale_symmetric_operator(laplacian)

    @staticmethod
    def rescale_symmetric_operator(matrix: np.ndarray) -> np.ndarray:
        """
        Rescale a real symmetric/Hermitian operator so that its spectrum is
        approximately mapped to [-1, 1]:

            M_rescaled = 2 * (M - lambda_min I) / (lambda_max - lambda_min) - I

        If lambda_max ~= lambda_min, a copy of the matrix is returned.

        Args:
            matrix: Square symmetric/Hermitian matrix.

        Returns:
            Rescaled matrix of the same shape.
        """
        matrix = np.asarray(matrix, dtype=float)

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square.")

        if not np.allclose(matrix, matrix.T.conj()):
            raise ValueError("matrix must be symmetric/Hermitian.")

        eigenvalues = np.linalg.eigvalsh(matrix)
        lambda_min = float(np.min(eigenvalues))
        lambda_max = float(np.max(eigenvalues))

        if np.isclose(lambda_max, lambda_min):
            return matrix.copy()

        identity = np.eye(matrix.shape[0], dtype=float)
        return 2.0 * (matrix - lambda_min * identity) / (lambda_max - lambda_min) - identity

    @staticmethod
    def normalize_by_spectral_radius(matrix: np.ndarray, atol: float = 1e-12) -> np.ndarray:
        """
        Normalize a symmetric/Hermitian matrix by its largest absolute eigenvalue.

        This is useful when only a bounded spectral radius is needed.

        Args:
            matrix: Square symmetric/Hermitian matrix.
            atol: Threshold below which the matrix is treated as spectrally zero.

        Returns:
            Normalized matrix of the same shape.
        """
        matrix = np.asarray(matrix, dtype=float)

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square.")

        if not np.allclose(matrix, matrix.T.conj()):
            raise ValueError("matrix must be symmetric/Hermitian.")

        eigenvalues = np.linalg.eigvalsh(matrix)
        radius = float(np.max(np.abs(eigenvalues)))

        if radius <= atol:
            return matrix.copy()

        return matrix / radius


def build_graph_input(
    graph: nx.Graph,
    node_signals: Optional[Sequence] = None,
    edge_signals: Optional[Dict[EdgeKey, Sequence]] = None,
) -> GraphInput:
    """
    Build a GraphInput object from a NetworkX graph.

    Args:
        graph: Input undirected graph.
        node_signals: Optional node features.
        edge_signals: Optional edge features.

    Returns:
        GraphInput instance.
    """
    node_order = sorted(graph.nodes())

    node_array = None
    if node_signals is not None:
        node_array = _as_2d_signal_array(node_signals, len(node_order), "node_signals")

    edge_dict: Dict[EdgeKey, np.ndarray] = {}
    if edge_signals is not None:
        for key, value in edge_signals.items():
            edge_dict[key] = np.asarray(value, dtype=float)

    return GraphInput(
        graph=graph,
        node_order=node_order,
        node_signals=node_array,
        edge_signals=edge_dict,
    )


def build_from_adjacency(
    adjacency: Union[np.ndarray, Sequence[Sequence[Number]]],
    node_signals: Optional[Sequence] = None,
    edge_signals: Optional[Dict[EdgeKey, Sequence]] = None,
) -> GraphInput:
    """
    Build a GraphInput object from an adjacency matrix.

    Args:
        adjacency: Square symmetric adjacency matrix.
        node_signals: Optional node features.
        edge_signals: Optional edge features.

    Returns:
        GraphInput instance.
    """
    adjacency = np.asarray(adjacency, dtype=float)

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Adjacency must be a square matrix.")

    if not np.allclose(adjacency, adjacency.T):
        raise ValueError("Adjacency must be symmetric.")

    graph = nx.from_numpy_array(adjacency)
    return build_graph_input(graph, node_signals=node_signals, edge_signals=edge_signals)


def default_two_node(
    node_signals: Optional[Sequence] = None,
    edge_signals: Optional[Dict[EdgeKey, Sequence]] = None,
) -> GraphInput:
    """Build the default two-node path graph."""
    return build_graph_input(nx.path_graph(2), node_signals=node_signals, edge_signals=edge_signals)


def default_path(
    num_nodes: int,
    node_signals: Optional[Sequence] = None,
    edge_signals: Optional[Dict[EdgeKey, Sequence]] = None,
) -> GraphInput:
    """Build a path graph with num_nodes nodes."""
    return build_graph_input(nx.path_graph(num_nodes), node_signals=node_signals, edge_signals=edge_signals)


def default_cycle(
    num_nodes: int,
    node_signals: Optional[Sequence] = None,
    edge_signals: Optional[Dict[EdgeKey, Sequence]] = None,
) -> GraphInput:
    """Build a cycle graph with num_nodes nodes."""
    return build_graph_input(nx.cycle_graph(num_nodes), node_signals=node_signals, edge_signals=edge_signals)


def default_complete(
    num_nodes: int,
    node_signals: Optional[Sequence] = None,
    edge_signals: Optional[Dict[EdgeKey, Sequence]] = None,
) -> GraphInput:
    """Build a complete graph with num_nodes nodes."""
    return build_graph_input(nx.complete_graph(num_nodes), node_signals=node_signals, edge_signals=edge_signals)


def default_star(
    num_nodes: int,
    node_signals: Optional[Sequence] = None,
    edge_signals: Optional[Dict[EdgeKey, Sequence]] = None,
) -> GraphInput:
    """
    Build a star graph with num_nodes nodes.

    Note:
        NetworkX star_graph(n) creates n + 1 nodes, so here we interpret
        num_nodes as the total number of nodes in the returned graph.
    """
    if num_nodes < 2:
        raise ValueError("A star graph must have at least 2 nodes.")

    graph = nx.star_graph(num_nodes - 1)
    return build_graph_input(graph, node_signals=node_signals, edge_signals=edge_signals)
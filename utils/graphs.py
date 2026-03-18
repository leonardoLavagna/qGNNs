#------------------------------------------------------------------------------
# graphs.py
#
# This module provides utility functions and data structures for handling graph
# inputs in a unified and validated way, including support for default graphs,
# custom graphs, node signals, and edge signals. Functions and classes included:
# - GraphInput: Container for a graph together with node and edge signals,
#   providing methods to compute adjacency, Laplacian, and normalized matrices.
# - build_graph_input: Creates a GraphInput
#   object from a NetworkX graph and optional signals.
# - build_from_adjacency: Builds a graph from an adjacency matrix 
#   and returns a GraphInput object.
# - default_two_node(...): Generates a simple two-node graph.
# - default_path(...): Generates a path graph with n nodes.
# - default_cycle(...): Generates a cycle graph with n nodes.
# - default_complete(...): Generates a complete graph with n nodes.
# - default_star(...): Generates a star graph with n nodes.
#
# These utilities are designed to provide a consistent interface for all
# downstream modules (classical GNN, quantum spectral filters, Chebyshev-based
# methods), ensuring that graphs and associated signals are validated and
# handled uniformly across the pipeline.
#
# © Leonardo Lavagna 2026
# leonardo.lavagna@uniroma1.it
#------------------------------------------------------------------------------


from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union
import networkx as nx
import numpy as np


Number = Union[int, float, complex]
EdgeKey = Tuple[int, int]


def _as_2d_signal_array(values: Sequence[Union[Number, Sequence[Number]]],
                        expected_length: int, name: str,) -> np.ndarray:
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
    for i, v in enumerate(values):
        arr = np.asarray(v, dtype=float)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        elif arr.ndim != 1:
            raise ValueError(f"{name}[{i}] must be scalar or 1D.")
        processed.append(arr)
    dims = {x.shape[0] for x in processed}
    if len(dims) != 1:
        raise ValueError(f"{name} must have consistent feature size.")

    return np.vstack(processed)


def _normalize_edge(u: int, v: int) -> Tuple[int, int]:
    """
    Normalize edge for undirected graph.

    Args:
        u: First node.
        v: Second node.

    Returns:
        Ordered edge tuple (min, max).
    """
    return (u, v) if u <= v else (v, u)


@dataclass
class GraphInput:
    """
    Container for graph + optional signals.

    Attributes:
        graph: NetworkX undirected graph.
        node_order: Ordered list of nodes.
        node_signals: Optional array (N, F_node).
        edge_signals: Dict mapping (u,v) -> feature vector.
    """
    graph: nx.Graph
    node_order: List[int]
    node_signals: Optional[np.ndarray] = None
    edge_signals: Dict[EdgeKey, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Run validation checks."""
        self._validate_graph()
        self._validate_node_order()
        self._validate_node_signals()
        self._validate_edge_signals()

    def _validate_graph(self) -> None:
        """Ensure graph is valid."""
        if self.graph.is_directed():
            raise ValueError("Only undirected graphs supported.")
        if self.graph.number_of_nodes() == 0:
            raise ValueError("Graph cannot be empty.")

    def _validate_node_order(self) -> None:
        """Ensure node_order matches graph nodes."""
        if set(self.node_order) != set(self.graph.nodes()):
            raise ValueError("node_order must match graph nodes.")

    def _validate_node_signals(self) -> None:
        """Validate node signals shape."""
        if self.node_signals is None:
            return
        arr = np.asarray(self.node_signals, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[0] != self.num_nodes:
            raise ValueError("node_signals must match number of nodes.")
        self.node_signals = arr

    def _validate_edge_signals(self) -> None:
        """Validate edge signals consistency."""
        validated = {}
        graph_edges = {
            _normalize_edge(int(u), int(v))
            for u, v in self.graph.edges()
        }
        feat_dim = None
        for (u, v), val in self.edge_signals.items():
            key = _normalize_edge(u, v)
            if key not in graph_edges:
                raise ValueError(f"Edge {key} not in graph.")
            arr = np.asarray(val, dtype=float)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            if feat_dim is None:
                feat_dim = arr.shape[0]
            elif arr.shape[0] != feat_dim:
                raise ValueError("Edge signals must have same dimension.")
            validated[key] = arr
        self.edge_signals = validated

    @property
    def num_nodes(self) -> int:
        """Number of nodes."""
        return len(self.node_order)

    def adjacency_matrix(self) -> np.ndarray:
        """
        Compute adjacency matrix.

        Returns:
            (N, N) adjacency matrix.
        """
        return nx.to_numpy_array(self.graph, nodelist=self.node_order)

    def degree_matrix(self) -> np.ndarray:
        """
        Compute degree matrix.

        Returns:
            (N, N) diagonal degree matrix.
        """
        A = self.adjacency_matrix()
        return np.diag(np.sum(A, axis=1))

    def laplacian_matrix(self) -> np.ndarray:
        """
        Compute Laplacian L = D - A.

        Returns:
            (N, N) Laplacian matrix.
        """
        return self.degree_matrix() - self.adjacency_matrix()

    def normalized_laplacian_matrix(self) -> np.ndarray:
        """
        Compute normalized Laplacian.

        Returns:
            (N, N) normalized Laplacian.
        """
        A = self.adjacency_matrix()
        d = np.sum(A, axis=1)

        if np.any(d == 0):
            raise ValueError("No isolated nodes allowed.")

        D_inv = np.diag(1.0 / np.sqrt(d))
        return np.eye(self.num_nodes) - D_inv @ A @ D_inv

    def kipf_welling_adjacency(self) -> np.ndarray:
        """
        Compute normalized adjacency with self-loops.

        Returns:
            (N, N) normalized adjacency.
        """
        A = self.adjacency_matrix()
        A_tilde = A + np.eye(self.num_nodes)
        d = np.sum(A_tilde, axis=1)
        D_inv = np.diag(1.0 / np.sqrt(d))
        return D_inv @ A_tilde @ D_inv


def build_graph_input(graph: nx.Graph, node_signals: Optional[Sequence] = None,
                      edge_signals: Optional[Dict[EdgeKey, Sequence]] = None,) -> GraphInput:
    """
    Build GraphInput from NetworkX graph.

    Args:
        graph: Input graph.
        node_signals: Optional node features.
        edge_signals: Optional edge features.

    Returns:
        GraphInput object.
    """
    node_order = sorted(graph.nodes())
    node_arr = None
    if node_signals is not None:
        node_arr = _as_2d_signal_array(node_signals, len(node_order), "node_signals")
    edge_dict = {}
    if edge_signals is not None:
        for k, v in edge_signals.items():
            edge_dict[k] = np.asarray(v, dtype=float)
    return GraphInput(
        graph=graph,
        node_order=node_order,
        node_signals=node_arr,
        edge_signals=edge_dict,
    )


def build_from_adjacency(adjacency: Union[np.ndarray, Sequence[Sequence[Number]]],
                         node_signals: Optional[Sequence] = None,
                         edge_signals: Optional[Dict[EdgeKey, Sequence]] = None,) -> GraphInput:
    """
    Build graph from adjacency matrix.

    Args:
        adjacency: Square symmetric matrix.
        node_signals: Optional node features.
        edge_signals: Optional edge features.

    Returns:
        GraphInput object.
    """
    A = np.asarray(adjacency, dtype=float)
    if A.shape[0] != A.shape[1]:
        raise ValueError("Adjacency must be square.")
    if not np.allclose(A, A.T):
        raise ValueError("Adjacency must be symmetric.")
    G = nx.from_numpy_array(A)
    return build_graph_input(G, node_signals, edge_signals)


def default_two_node(node_signals: Optional[Sequence] = None,
                     edge_signals: Optional[Dict[EdgeKey, Sequence]] = None,) -> GraphInput:
    """
    Create 2-node graph.

    Returns:
        GraphInput.
    """
    return build_graph_input(nx.path_graph(2), node_signals, edge_signals)


def default_path(n: int, node_signals=None, edge_signals=None) -> GraphInput:
    """Create path graph."""
    return build_graph_input(nx.path_graph(n), node_signals, edge_signals)


def default_cycle(n: int, node_signals=None, edge_signals=None) -> GraphInput:
    """Create cycle graph."""
    return build_graph_input(nx.cycle_graph(n), node_signals, edge_signals)


def default_complete(n: int, node_signals=None, edge_signals=None) -> GraphInput:
    """Create complete graph."""
    return build_graph_input(nx.complete_graph(n), node_signals, edge_signals)


def default_star(n: int, node_signals=None, edge_signals=None) -> GraphInput:
    """Create star graph."""
    return build_graph_input(nx.star_graph(n - 1), node_signals, edge_signals)


DEFAULT_GRAPHS = {
    "two_node": default_two_node,
    "path": default_path,
    "cycle": default_cycle,
    "complete": default_complete,
    "star": default_star,
}
#------------------------------------------------------------------------------
# quantum_graph_layers.py
#
# This module implements minimal quantum graph neural network layers compatible
# with the existing GraphInput pipeline. It provides a quantum extension of the
# classical Kipf–Welling framework by replacing the propagation term A_hat @ X
# with a graph-dependent quantum evolution of the form exp(-i * alpha * L_tilde) 
# applied to X where: A_hat is the Kipf–Welling normalized adjacency, and 
# L_tilde = I - A_hat is the associated Laplacian-like operator. Functions and 
# classes included:
# - BaseQuantumGraphFilter: Abstract interface for graph-dependent quantum propagation.
# - ExponentialQuantumGraphFilter: Minimal quantum spectral filter based on exp(-i * alpha * L_tilde),
#   implemented via statevector simulation (Qiskit).
# - QuantumGraphLayer: Single quantum graph layer mirroring the classical structure
#   H_out = activation(Q_G(H_in) @ W).
# - SingleLayerQuantumGraphNetwork: Minimal single-layer quantum graph neural network.
# - TwoLayerQuantumGraphNetwork: Minimal two-layer quantum graph neural network.
# The module is designed to:
# - reuse GraphInput without introducing new data structures
# - preserve the forward interface: forward(graph_input, features=None) -> np.ndarray
# - act as a drop-in replacement for the classical propagation step
# This implementation is intentionally minimal and targets small graphs using
# statevector simulation. It provides a foundation for future extensions toward
# general spectral filters g_theta(L)
#
# © Leonardo Lavagna 2026
# leonardo.lavagna@uniroma1.it
#------------------------------------------------------------------------------


from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Optional
import numpy as np
from qiskit.quantum_info import Operator, Statevector
from scipy.linalg import expm
from classical_gnns.shallow_kipf_welling_gnn import relu, identity   


class BaseQuantumGraphFilter(ABC):
    """Base interface for quantum graph propagation.

    This is the quantum analogue of the classical graph propagation block,
    such as A_hat @ X in a shallow Kipf-Welling layer.

    Implementations must:
    - read graph-dependent information from an existing GraphInput instance
    - preserve the feature matrix shape (N, F)
    - return a NumPy array
    """

    @abstractmethod
    def forward(self,graph_input,features: np.ndarray,) -> np.ndarray:
        """Apply graph-dependent quantum propagation.

        Args:
            graph_input: Existing GraphInput instance.
            features: Feature matrix of shape (N, F).

        Returns:
            Propagated feature matrix of shape (N, F).
        """
        raise NotImplementedError


class ExponentialQuantumGraphFilter(BaseQuantumGraphFilter):
    """Quantum spectral filter based on exp(-i * alpha * L_tilde).

    The graph-dependent operator is built from the Kipf-Welling normalized
    adjacency already available through GraphInput:

        A_hat = graph_input.kipf_welling_adjacency()
        L_tilde = I - A_hat

    The filter applies the unitary evolution

        U = exp(-i * alpha * L_tilde_rescaled)

    to each feature column independently via amplitude encoding.

    Notes:
        - This is a minimal statevector-based implementation.
        - Small graphs are assumed.
        - For graphs whose number of nodes is not a power of two, padding
          is applied internally.
        - The decoded output is taken as the real part of the evolved state
          amplitudes, rescaled by the original input norm.
    """

    def __init__(self,alpha: float = 1.0,rescale_laplacian: bool = True,) -> None:
        """Initialize the exponential quantum graph filter.

        Args:
            alpha: Evolution strength in exp(-i * alpha * L_tilde).
            rescale_laplacian: Whether to rescale the Laplacian spectrum
                to approximately lie in [-1, 1] before exponentiation.
        """
        self.alpha = alpha
        self.rescale_laplacian = rescale_laplacian

    def forward(self,graph_input,features: np.ndarray,) -> np.ndarray:
        """Apply the exponential quantum spectral filter.

        Args:
            graph_input: Existing GraphInput instance.
            features: Input features of shape (N, F).

        Returns:
            Output features of shape (N, F).
        """
        features = np.asarray(features, dtype=float)
        if features.ndim != 2:
            raise ValueError(
                "features must have shape (N, F). "
                f"Received array with shape {features.shape}."
            )
        laplacian = self._build_laplacian(graph_input)
        if self.rescale_laplacian:
            laplacian = self._rescale_hermitian(laplacian)
        unitary = self._build_unitary(laplacian)
        outputs = []
        for feature_idx in range(features.shape[1]):
            column = features[:, feature_idx]
            propagated = self._apply_unitary_to_feature(column, unitary)
            outputs.append(propagated)
        return np.stack(outputs, axis=1)

    def _build_laplacian(self, graph_input) -> np.ndarray:
        """Build the Laplacian used by the quantum filter.

        This implementation uses the graph operator most directly aligned
        with the current classical Kipf-Welling pipeline:

            L_tilde = I - A_hat

        where A_hat is the normalized adjacency already computed inside
        GraphInput.

        Args:
            graph_input: Existing GraphInput instance.

        Returns:
            Laplacian-like matrix of shape (N, N).
        """
        adjacency_normalized = np.asarray(graph_input.kipf_welling_adjacency(),dtype=float,)
        if adjacency_normalized.ndim != 2:
            raise ValueError(
                "graph_input.kipf_welling_adjacency() must be a matrix."
            )
        num_nodes = adjacency_normalized.shape[0]
        identity = np.eye(num_nodes, dtype=float)
        laplacian = identity - adjacency_normalized
        return laplacian

    def _rescale_hermitian(self, matrix: np.ndarray) -> np.ndarray:
        """Rescale a symmetric/Hermitian matrix to a stable spectral range.

        For small graphs we can explicitly estimate the extremal eigenvalues
        and map the spectrum approximately to [-1, 1]:

            M_rescaled = 2 * (M - lambda_min I) / (lambda_max - lambda_min) - I

        If the matrix is numerically constant in spectrum, it is returned
        unchanged.

        Args:
            matrix: Real symmetric matrix.

        Returns:
            Rescaled matrix of the same shape.
        """
        eigenvalues = np.linalg.eigvalsh(matrix)
        lambda_min = float(np.min(eigenvalues))
        lambda_max = float(np.max(eigenvalues))
        if np.isclose(lambda_max, lambda_min):
            return matrix.copy()
        identity_ = np.eye(matrix.shape[0], dtype=float)
        rescaled = (2.0 * (matrix - lambda_min * identity_) / (lambda_max - lambda_min)- identity_)
        return rescaled

    def _build_unitary(self, laplacian: np.ndarray) -> np.ndarray:
        """Build the padded unitary matrix exp(-i * alpha * L).

        Args:
            laplacian: Real symmetric matrix of shape (N, N).

        Returns:
            Complex unitary matrix of shape (2^n, 2^n), where 2^n >= N.
        """
        padded_laplacian = self._pad_square_matrix_to_power_of_two(laplacian)
        unitary = expm(-1j * self.alpha * padded_laplacian)
        return unitary

    def _apply_unitary_to_feature(self,feature_vector: np.ndarray,unitary: np.ndarray,) -> np.ndarray:
        """Apply the graph-dependent unitary to one feature channel.

        Args:
            feature_vector: Real vector of shape (N,).
            unitary: Complex unitary matrix of shape (2^n, 2^n).

        Returns:
            Real propagated vector of shape (N,).
        """
        feature_vector = np.asarray(feature_vector, dtype=float)
        if feature_vector.ndim != 1:
            raise ValueError(
                "feature_vector must have shape (N,). "
                f"Received array with shape {feature_vector.shape}.")
        original_norm = float(np.linalg.norm(feature_vector))
        if np.isclose(original_norm, 0.0):
            return feature_vector.copy()
        normalized = feature_vector / original_norm
        padded_state = self._pad_vector_to_power_of_two(normalized)
        state = Statevector(padded_state)
        evolved_state = state.evolve(Operator(unitary))
        num_nodes = feature_vector.shape[0]
        decoded = np.real(evolved_state.data[:num_nodes])
        return original_norm * decoded

    def _pad_vector_to_power_of_two(self, vector: np.ndarray) -> np.ndarray:
        """Pad a vector with zeros so that its length is a power of two.

        Args:
            vector: Input vector of shape (N,).

        Returns:
            Padded vector of shape (2^n,).
        """
        size = vector.shape[0]
        target_size = self._next_power_of_two(size)
        if target_size == size:
            return vector.astype(complex)
        padded = np.zeros(target_size, dtype=complex)
        padded[:size] = vector
        return padded

    def _pad_square_matrix_to_power_of_two(self, matrix: np.ndarray) -> np.ndarray:
        """Pad a square matrix with zeros so that its size is a power of two.

        Args:
            matrix: Input matrix of shape (N, N).

        Returns:
            Padded matrix of shape (2^n, 2^n).
        """
        rows, cols = matrix.shape
        if rows != cols:
            raise ValueError(
                "matrix must be square. "
                f"Received shape {matrix.shape}.")
        target_size = self._next_power_of_two(rows)
        if target_size == rows:
            return matrix.astype(complex)
        padded = np.zeros((target_size, target_size), dtype=complex)
        padded[:rows, :cols] = matrix
        return padded

    @staticmethod
    def _next_power_of_two(value: int) -> int:
        """Return the smallest power of two greater than or equal to value.

        Args:
            value: Positive integer.

        Returns:
            Smallest power of two >= value.
        """
        if value <= 1:
            return 1
        return 1 << (value - 1).bit_length()


class QuantumGraphLayer:
    """Quantum analogue of the shallow Kipf-Welling layer.

    This layer preserves the same interface used by the classical modules:

        forward(graph_input, features=None) -> np.ndarray

    and mirrors the classical structure:

        classical: H_out = activation(A_hat @ H_in @ W)
        quantum:   H_out = activation(Q_G(H_in) @ W)

    where Q_G is a graph-dependent quantum propagation operator implemented
    by a BaseQuantumGraphFilter.
    """

    def __init__(self,in_features: int,out_features: int,
                 quantum_filter: Optional[BaseQuantumGraphFilter] = None,
                 activation: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 use_bias: bool = True,random_state: Optional[int] = 0,) -> None:
        """Initialize the quantum graph layer.

        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            quantum_filter: Quantum propagation block. If None, a default
                ExponentialQuantumGraphFilter is used.
            activation: Optional activation function.
            use_bias: Whether to use an additive bias term.
            random_state: Seed for deterministic weight initialization.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.quantum_filter = (
            quantum_filter
            if quantum_filter is not None
            else ExponentialQuantumGraphFilter())
        self.activation = activation if activation is not None else identity
        self.use_bias = use_bias
        rng = np.random.default_rng(random_state)
        self.weight = 0.1 * rng.standard_normal((in_features, out_features))
        if self.use_bias:
            self.bias = np.zeros(out_features, dtype=float)
        else:
            self.bias = None

    def forward(self,graph_input,features: Optional[np.ndarray] = None,) -> np.ndarray:
        """Apply quantum propagation, linear mixing, and activation.

        Args:
            graph_input: Existing GraphInput instance.
            features: Optional input features of shape (N, F_in). If None,
                graph_input.node_signals is used.

        Returns:
            Output feature matrix of shape (N, F_out).
        """
        if features is None:
            features = graph_input.node_signals
        features = np.asarray(features, dtype=float)
        if features.ndim != 2:
            raise ValueError(
                "features must have shape (N, F_in). "
                f"Received array with shape {features.shape}."
            )
        if features.shape[1] != self.in_features:
            raise ValueError(
                "features.shape[1] must match in_features. "
                f"Received features.shape[1]={features.shape[1]} "
                f"and in_features={self.in_features}.")
        propagated = self.quantum_filter.forward(graph_input, features)
        output = propagated @ self.weight
        if self.bias is not None:
            output = output + self.bias
        output = self.activation(output)
        return output


class SingleLayerQuantumGraphNetwork:
    """Minimal single-layer quantum graph neural network.

    This class is a thin wrapper around QuantumGraphLayer and provides
    a network-style interface consistent with the classical modules.
    It is useful when the whole model consists of a single quantum
    graph propagation layer followed by linear mixing and activation.
    """

    def __init__(self,in_features: int,out_features: int,
                 quantum_filter: Optional[BaseQuantumGraphFilter] = None,
                 activation: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 use_bias: bool = True,random_state: Optional[int] = 0,) -> None:
        """Initialize the single-layer quantum graph network.

        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            quantum_filter: Quantum propagation block. If None, a default
                ExponentialQuantumGraphFilter is used.
            activation: Optional activation function.
            use_bias: Whether to use an additive bias term.
            random_state: Seed for deterministic initialization.
        """
        self.layer = QuantumGraphLayer(
            in_features=in_features,
            out_features=out_features,
            quantum_filter=quantum_filter,
            activation=activation,
            use_bias=use_bias,
            random_state=random_state,
        )

    def forward( self, graph_input, features: Optional[np.ndarray] = None,) -> np.ndarray:
        """Apply the single-layer quantum graph network.

        Args:
            graph_input: Existing GraphInput instance.
            features: Optional input features. If None, graph_input.node_signals
                is used.

        Returns:
            Output feature matrix of shape (N, out_features).
        """
        return self.layer.forward(graph_input, features)


class TwoLayerQuantumGraphNetwork:
    """Minimal two-layer quantum graph network.

    This mirrors the simple two-layer classical GNN already present in the
    project, while keeping the exact same forward interface.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        first_filter: Optional[BaseQuantumGraphFilter] = None,
        second_filter: Optional[BaseQuantumGraphFilter] = None,
        hidden_activation: Optional[Callable[[np.ndarray], np.ndarray]] = relu,
        output_activation: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        random_state: Optional[int] = 0,
    ) -> None:
        """Initialize the two-layer quantum graph network.

        Args:
            in_features: Input feature dimension.
            hidden_features: Hidden feature dimension.
            out_features: Output feature dimension.
            first_filter: Quantum filter for the first layer.
            second_filter: Quantum filter for the second layer.
            hidden_activation: Activation after the first layer.
            output_activation: Activation after the second layer.
            random_state: Seed for deterministic initialization.
        """
        first_seed = random_state
        second_seed = None if random_state is None else random_state + 1

        self.layer1 = QuantumGraphLayer(
            in_features=in_features,
            out_features=hidden_features,
            quantum_filter=first_filter,
            activation=hidden_activation,
            random_state=first_seed,
        )
        self.layer2 = QuantumGraphLayer(
            in_features=hidden_features,
            out_features=out_features,
            quantum_filter=second_filter,
            activation=output_activation,
            random_state=second_seed,
        )

    def forward( self, graph_input, features: Optional[np.ndarray] = None,) -> np.ndarray:
        """Apply the two-layer quantum graph network.

        Args:
            graph_input: Existing GraphInput instance.
            features: Optional input features. If None, graph_input.node_signals
                is used.

        Returns:
            Output feature matrix of shape (N, out_features).
        """
        hidden = self.layer1.forward(graph_input, features)
        output = self.layer2.forward(graph_input, hidden)
        return output
    
class FirstOrderQuantumGraphFilter(BaseQuantumGraphFilter): # NEW
    """Minimal first-order spectral graph filter. This implements a graph filter of the form

        g_theta(X) = theta_0 * X + theta_1 * A_hat @ X

    where A_hat is the Kipf-Welling normalized adjacency already available
    through GraphInput.

    Notes:
        - This is a quantum-inspired / hybrid-ready spectral filter.
        - The coefficients can later be generated by a quantum circuit without
          changing the forward interface.
    """

    def __init__(self,theta_0: float = 1.0,theta_1: float = 1.0,) -> None:
        self.theta_0 = float(theta_0)
        self.theta_1 = float(theta_1)

    def forward(self, graph_input, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=float)
        if features.ndim != 2:
            raise ValueError(
                "features must have shape (N, F). "
                f"Received array with shape {features.shape}."
            )
        a_hat = np.asarray(graph_input.kipf_welling_adjacency(), dtype=float)
        if a_hat.shape[0] != features.shape[0]:
            raise ValueError(
                "Mismatch between graph size and feature rows. "
                f"Received A_hat.shape={a_hat.shape} and features.shape={features.shape}."
            )
        return self.theta_0 * features + self.theta_1 * (a_hat @ features)
    

class ChebyshevQuantumGraphFilter(BaseQuantumGraphFilter): # NEW
    """Chebyshev spectral graph filter. Implements
        
        g_theta(L_tilde) X = sum_{k=0}^K theta_k T_k(L_tilde) X
    
    where:
        A_hat   = graph_input.kipf_welling_adjacency()
        L_tilde = I - A_hat
    """

    def __init__(self,coefficients: list[float] | np.ndarray,rescale_laplacian: bool = True,) -> None:
        coeffs = np.asarray(coefficients, dtype=float)
        if coeffs.ndim != 1 or coeffs.size == 0:
            raise ValueError("coefficients must be a non-empty 1D array.")
        self.coefficients = coeffs
        self.rescale_laplacian = rescale_laplacian

    def forward(self, graph_input, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=float)
        if features.ndim != 2:
            raise ValueError(
                "features must have shape (N, F). "
                f"Received array with shape {features.shape}."
            )

        a_hat = np.asarray(graph_input.kipf_welling_adjacency(), dtype=float)
        n = a_hat.shape[0]
        if features.shape[0] != n:
            raise ValueError(
                "Mismatch between graph size and feature rows. "
                f"Received A_hat.shape={a_hat.shape} and features.shape={features.shape}."
            )

        laplacian = np.eye(n, dtype=float) - a_hat
        if self.rescale_laplacian:
            laplacian = self._rescale_hermitian(laplacian)

        return self._apply_chebyshev_filter(laplacian, features)

    def _apply_chebyshev_filter(self,operator_matrix: np.ndarray,features: np.ndarray,) -> np.ndarray:
        coeffs = self.coefficients
        k_max = coeffs.size - 1
        t0 = features
        output = coeffs[0] * t0
        if k_max == 0:
            return output
        t1 = operator_matrix @ features
        output = output + coeffs[1] * t1
        for _k in range(2, k_max + 1):
            t2 = 2.0 * (operator_matrix @ t1) - t0
            output = output + coeffs[_k] * t2
            t0, t1 = t1, t2
        return output
    
    @staticmethod
    def _rescale_hermitian(matrix: np.ndarray) -> np.ndarray:
        eigenvalues = np.linalg.eigvalsh(matrix)
        lambda_max = np.max(np.abs(eigenvalues))
        if lambda_max <= 1e-12:
            return matrix.copy()
        return matrix / lambda_max
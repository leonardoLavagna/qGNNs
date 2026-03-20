# -----------------------------------------------------------------------------
# qgtheta.py
#
# Quantum-ready spectral graph filters and hybrid quantum-classical graph layers
# compatible with the existing GraphInput pipeline.
#
# This module provides graph-dependent propagation blocks that can act as
# drop-in replacements for classical graph propagation steps in shallow or
# multi-layer GNN architectures. It reuses GraphInput as the common graph
# container and assumes that graph-derived operators such as:
#
#     A_hat   = Kipf-Welling normalized adjacency
#     L_tilde = I - A_hat
#
# are constructed in utils.graphs through GraphInput.
#
# Main responsibilities:
# - define abstract interfaces for graph-dependent propagation blocks
# - define abstract interfaces for spectral coefficient generators
# - implement a hybrid Qiskit-based variational coefficient provider
# - implement exponential quantum Laplacian evolution as an exploratory model
# - implement first-order spectral filtering based on A_hat
# - implement Chebyshev polynomial spectral filtering
# - implement a block-encoding-inspired polynomial Laplacian filter
# - implement a mimic/approximation route based on quantum evolution mixtures
# - support dynamic, quantum-generated spectral coefficients
# - define graph layers with propagation, linear mixing, and activation
# - define minimal one-layer and two-layer graph network wrappers
#
# Classes:
# - BaseQuantumGraphFilter:
#   Abstract base class for graph-dependent propagation blocks acting on
#   feature matrices.
# - BaseQuantumCoefficientProvider:
#   Abstract base class for modules that generate spectral filter
#   coefficients from graph and feature context.
# - BaseQuantumOperatorModel:
#   Abstract base class for quantum-operator-inspired models that act on a
#   graph operator and node-feature matrix.
# - QiskitVariationalCoefficientProvider:
#   Hybrid quantum-classical controller that uses a small Qiskit circuit to
#   generate bounded spectral coefficients from graph and feature summaries.
# - ExponentialQuantumGraphFilter:
#   Exploratory quantum graph propagator based on the unitary evolution
#   exp(-i * alpha * L_tilde), applied independently to each feature channel
#   via amplitude encoding and statevector simulation.
# - FirstOrderQuantumGraphFilter:
#   First-order spectral graph filter of the form
#       theta_0 * X + theta_1 * A_hat @ X,
#   optionally with dynamically generated coefficients.
# - ChebyshevQuantumGraphFilter:
#   Spectral graph filter based on a truncated Chebyshev expansion
#       sum_k theta_k T_k(L_tilde) X,
#   optionally with dynamically generated coefficients.
# - PolynomialBlockEncodingQuantumGraphFilter:
#   Block-encoding-inspired spectral filter that treats g_theta(L_tilde) as
#   a polynomial in L_tilde and applies it through a simulated polynomial
#   operator pipeline.
# - MimicQuantumGraphFilter:
#   Approximate quantum graph filter that mimics g_theta(L_tilde) using a
#   weighted combination of graph-aware quantum evolutions and optional
#   residual terms.
# - QuantumGraphLayer:
#   Single graph layer implementing the pattern
#       activation(Q_G(H_in) @ W + b).
# - SingleLayerQuantumGraphNetwork:
#   Thin wrapper around one QuantumGraphLayer.
# - TwoLayerQuantumGraphNetwork:
#   Minimal two-layer graph neural network composed of two QuantumGraphLayer
#   instances.
#
# © Leonardo Lavagna 2026
# leonardo.lavagna@uniroma1.it
# -----------------------------------------------------------------------------
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Optional
import numpy as np
from scipy.linalg import expm
from classical_gnns.shallow_kipf_welling_gnn import identity, relu
from utils.graphs import GraphInput
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector


class BaseQuantumGraphFilter(ABC):
    @abstractmethod
    def forward(self, graph_input: GraphInput, features: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class BaseQuantumCoefficientProvider(ABC):
    @abstractmethod
    def get_coefficients(
        self,
        graph_input: GraphInput,
        features: np.ndarray,
        num_coefficients: int,
    ) -> np.ndarray:
        raise NotImplementedError


class BaseQuantumOperatorModel(ABC):
    """
    Abstract interface for operator-based quantum graph models.

    Implementations receive a graph operator (typically a Laplacian-like matrix)
    together with a feature matrix and return a propagated feature matrix.
    """

    @abstractmethod
    def apply(
        self,
        operator_matrix: np.ndarray,
        features: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError


class QiskitVariationalCoefficientProvider(BaseQuantumCoefficientProvider):
    """
    Trainable Qiskit-based coefficient generator.

    The provider builds a compact context vector from the current graph and
    features, embeds it into a shallow quantum circuit, evaluates expectation
    values, and maps them to spectral coefficients.

    Parameters are trainable through an external classical optimizer.
    """

    def __init__(
        self,
        num_qubits: int = 2,
        num_layers: int = 1,
        input_scale: float = 0.2,
        output_scale: float = 0.1,
        output_bias: Optional[np.ndarray] = None,
        trainable_weights: Optional[np.ndarray] = None,
        normalize_coefficients: bool = False,
        normalization_eps: float = 1e-12,
    ) -> None:
        if num_qubits < 1:
            raise ValueError("num_qubits must be at least 1.")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.num_qubits = int(num_qubits)
        self.num_layers = int(num_layers)
        self.input_scale = float(input_scale)
        self.output_scale = float(output_scale)
        self.normalize_coefficients = bool(normalize_coefficients)
        self.normalization_eps = float(normalization_eps)

        if output_bias is None:
            self.output_bias = None
        else:
            self.output_bias = np.asarray(output_bias, dtype=float)

        expected_shape = (self.num_layers, self.num_qubits, 2)
        if trainable_weights is None:
            self.trainable_weights = np.zeros(expected_shape, dtype=float)
        else:
            weights = np.asarray(trainable_weights, dtype=float)
            if weights.shape != expected_shape:
                raise ValueError(
                    f"trainable_weights must have shape {expected_shape}. "
                    f"Received {weights.shape}."
                )
            self.trainable_weights = weights

    def get_parameter_vector(self) -> np.ndarray:
        """Return the trainable circuit parameters as a flat vector."""
        return self.trainable_weights.reshape(-1).copy()

    def set_parameter_vector(self, parameter_vector: np.ndarray) -> None:
        """Overwrite the trainable circuit parameters from a flat vector."""
        parameter_vector = np.asarray(parameter_vector, dtype=float)
        expected_size = self.num_layers * self.num_qubits * 2
        if parameter_vector.size != expected_size:
            raise ValueError(
                f"parameter_vector must have size {expected_size}. "
                f"Received size {parameter_vector.size}."
            )
        self.trainable_weights = parameter_vector.reshape(
            self.num_layers,
            self.num_qubits,
            2,
        ).copy()

    def num_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return self.num_layers * self.num_qubits * 2

    def get_coefficients(
        self,
        graph_input: GraphInput,
        features: np.ndarray,
        num_coefficients: int,
    ) -> np.ndarray:
        features = np.asarray(features, dtype=float)
        if features.ndim != 2:
            raise ValueError(
                "features must have shape (N, F). "
                f"Received shape {features.shape}."
            )
        if num_coefficients < 1:
            raise ValueError("num_coefficients must be at least 1.")

        context = self._build_context_vector(graph_input, features)
        circuit = self._build_circuit(context)
        state = Statevector.from_instruction(circuit)
        raw_values = self._measure_observables(state, num_coefficients)

        if self.output_bias is None:
            bias = np.zeros(num_coefficients, dtype=float)
        else:
            if self.output_bias.size < num_coefficients:
                raise ValueError(
                    "output_bias does not contain enough entries for the requested "
                    f"{num_coefficients} coefficients."
                )
            bias = self.output_bias[:num_coefficients]

        coefficients = bias + self.output_scale * raw_values

        if self.normalize_coefficients:
            denom = np.sum(np.abs(coefficients))
            if denom > self.normalization_eps:
                coefficients = coefficients / denom

        return coefficients

    def _build_context_vector(
        self,
        graph_input: GraphInput,
        features: np.ndarray,
    ) -> np.ndarray:
        a_hat = np.asarray(graph_input.kipf_welling_adjacency(), dtype=float)
        num_nodes = float(graph_input.num_nodes)

        degree_proxy = np.sum(a_hat, axis=1)
        feature_norms = np.linalg.norm(features, axis=1)

        context = np.array(
            [
                np.mean(features),
                np.std(features),
                np.mean(np.abs(features)),
                np.linalg.norm(features) / np.sqrt(features.size),
                np.mean(degree_proxy),
                np.std(degree_proxy),
                np.trace(a_hat) / max(num_nodes, 1.0),
                np.mean(feature_norms),
            ],
            dtype=float,
        )
        return context

    def _build_circuit(self, context: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        repeated_context = np.resize(context, self.num_qubits)

        for layer in range(self.num_layers):
            for qubit in range(self.num_qubits):
                data_angle = self.input_scale * repeated_context[qubit]
                ry_offset = self.trainable_weights[layer, qubit, 0]
                rz_offset = self.trainable_weights[layer, qubit, 1]

                qc.ry(data_angle + ry_offset, qubit)
                qc.rz(data_angle + rz_offset, qubit)

            for qubit in range(self.num_qubits - 1):
                qc.cx(qubit, qubit + 1)

            if self.num_qubits > 2:
                qc.cx(self.num_qubits - 1, 0)

        return qc

    def _measure_observables(
        self,
        state: Statevector,
        num_coefficients: int,
    ) -> np.ndarray:
        values: list[float] = []

        for qubit in range(self.num_qubits):
            values.append(self._expectation_single_pauli(state, "Z", qubit))

        for qubit in range(self.num_qubits - 1):
            values.append(
                self._expectation_two_pauli(state, "Z", qubit, "Z", qubit + 1)
            )

        if self.num_qubits > 2:
            values.append(
                self._expectation_two_pauli(
                    state,
                    "Z",
                    self.num_qubits - 1,
                    "Z",
                    0,
                )
            )

        for qubit in range(self.num_qubits):
            values.append(self._expectation_single_pauli(state, "X", qubit))

        values_array = np.asarray(values, dtype=float)
        if values_array.size < num_coefficients:
            values_array = np.resize(values_array, num_coefficients)

        return values_array[:num_coefficients]

    def _expectation_single_pauli(
        self,
        state: Statevector,
        pauli_label: str,
        qubit: int,
    ) -> float:
        label = ["I"] * self.num_qubits
        label[self.num_qubits - 1 - qubit] = pauli_label
        operator = SparsePauliOp("".join(label))
        value = state.expectation_value(operator)
        return float(np.real(value))

    def _expectation_two_pauli(
        self,
        state: Statevector,
        pauli_a: str,
        qubit_a: int,
        pauli_b: str,
        qubit_b: int,
    ) -> float:
        label = ["I"] * self.num_qubits
        label[self.num_qubits - 1 - qubit_a] = pauli_a
        label[self.num_qubits - 1 - qubit_b] = pauli_b
        operator = SparsePauliOp("".join(label))
        value = state.expectation_value(operator)
        return float(np.real(value))


class ExponentialQuantumGraphFilter(BaseQuantumGraphFilter):
    def __init__(self, alpha: float = 1.0, rescale_laplacian: bool = True) -> None:
        self.alpha = float(alpha)
        self.rescale_laplacian = rescale_laplacian

    def forward(self, graph_input: GraphInput, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=float)
        if features.ndim != 2:
            raise ValueError(
                "features must have shape (N, F). "
                f"Received shape {features.shape}."
            )

        if features.shape[0] != graph_input.num_nodes:
            raise ValueError(
                "Mismatch between graph size and feature rows. "
                f"Received graph_input.num_nodes={graph_input.num_nodes} "
                f"and features.shape={features.shape}."
            )

        laplacian = self._get_laplacian(graph_input)
        unitary = self._build_unitary(laplacian)

        outputs = []
        for feature_index in range(features.shape[1]):
            feature_column = features[:, feature_index]
            propagated = self._apply_unitary_to_feature(feature_column, unitary)
            outputs.append(propagated)

        return np.stack(outputs, axis=1)

    def _get_laplacian(self, graph_input: GraphInput) -> np.ndarray:
        if self.rescale_laplacian:
            return graph_input.rescaled_kipf_welling_laplacian()
        return graph_input.kipf_welling_laplacian()

    def _build_unitary(self, laplacian: np.ndarray) -> np.ndarray:
        padded_laplacian = self._pad_square_matrix_to_power_of_two(laplacian)
        return expm(-1j * self.alpha * padded_laplacian)

    def _apply_unitary_to_feature(
        self,
        feature_vector: np.ndarray,
        unitary: np.ndarray,
    ) -> np.ndarray:
        feature_vector = np.asarray(feature_vector, dtype=float)
        if feature_vector.ndim != 1:
            raise ValueError(
                "feature_vector must have shape (N,). "
                f"Received shape {feature_vector.shape}."
            )

        original_norm = float(np.linalg.norm(feature_vector))
        if np.isclose(original_norm, 0.0):
            return feature_vector.copy()

        normalized = feature_vector / original_norm
        padded_state = self._pad_vector_to_power_of_two(normalized)
        evolved_state = Statevector(padded_state).evolve(Operator(unitary))

        num_nodes = feature_vector.shape[0]
        decoded = np.real(evolved_state.data[:num_nodes])
        return original_norm * decoded

    def _pad_vector_to_power_of_two(self, vector: np.ndarray) -> np.ndarray:
        size = vector.shape[0]
        target_size = self._next_power_of_two(size)
        if target_size == size:
            return vector.astype(complex)

        padded = np.zeros(target_size, dtype=complex)
        padded[:size] = vector
        return padded

    def _pad_square_matrix_to_power_of_two(self, matrix: np.ndarray) -> np.ndarray:
        rows, cols = matrix.shape
        if rows != cols:
            raise ValueError(f"matrix must be square. Received shape {matrix.shape}.")

        target_size = self._next_power_of_two(rows)
        if target_size == rows:
            return matrix.astype(complex)

        padded = np.zeros((target_size, target_size), dtype=complex)
        padded[:rows, :cols] = matrix
        return padded

    @staticmethod
    def _next_power_of_two(value: int) -> int:
        if value <= 1:
            return 1
        return 1 << (value - 1).bit_length()


class FirstOrderQuantumGraphFilter(BaseQuantumGraphFilter):
    def __init__(
        self,
        theta_0: float = 1.0,
        theta_1: float = 1.0,
        coefficient_provider: Optional[BaseQuantumCoefficientProvider] = None,
    ) -> None:
        self.theta_0 = float(theta_0)
        self.theta_1 = float(theta_1)
        self.coefficient_provider = coefficient_provider

    def forward(self, graph_input: GraphInput, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=float)
        if features.ndim != 2:
            raise ValueError(
                "features must have shape (N, F). "
                f"Received shape {features.shape}."
            )

        a_hat = np.asarray(graph_input.kipf_welling_adjacency(), dtype=float)
        if a_hat.shape[0] != features.shape[0]:
            raise ValueError(
                "Mismatch between graph size and feature rows. "
                f"Received A_hat.shape={a_hat.shape} and features.shape={features.shape}."
            )

        if self.coefficient_provider is None:
            theta_0 = self.theta_0
            theta_1 = self.theta_1
        else:
            coeffs = self.coefficient_provider.get_coefficients(
                graph_input=graph_input,
                features=features,
                num_coefficients=2,
            )
            theta_0 = float(coeffs[0])
            theta_1 = float(coeffs[1])

        return theta_0 * features + theta_1 * (a_hat @ features)


class ChebyshevQuantumGraphFilter(BaseQuantumGraphFilter):
    def __init__(
        self,
        coefficients: list[float] | np.ndarray,
        rescale_laplacian: bool = True,
        coefficient_provider: Optional[BaseQuantumCoefficientProvider] = None,
    ) -> None:
        coeffs = np.asarray(coefficients, dtype=float)
        if coeffs.ndim != 1 or coeffs.size == 0:
            raise ValueError("coefficients must be a non-empty 1D array.")

        self.coefficients = coeffs
        self.rescale_laplacian = rescale_laplacian
        self.coefficient_provider = coefficient_provider

    def forward(self, graph_input: GraphInput, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=float)
        if features.ndim != 2:
            raise ValueError(
                "features must have shape (N, F). "
                f"Received shape {features.shape}."
            )

        if features.shape[0] != graph_input.num_nodes:
            raise ValueError(
                "Mismatch between graph size and feature rows. "
                f"Received graph_input.num_nodes={graph_input.num_nodes} "
                f"and features.shape={features.shape}."
            )

        if self.coefficient_provider is None:
            coefficients = self.coefficients
        else:
            coefficients = self.coefficient_provider.get_coefficients(
                graph_input=graph_input,
                features=features,
                num_coefficients=self.coefficients.size,
            )

        if self.rescale_laplacian:
            laplacian = graph_input.rescaled_kipf_welling_laplacian()
        else:
            laplacian = graph_input.kipf_welling_laplacian()

        return self._apply_chebyshev_filter(laplacian, features, coefficients)

    def _apply_chebyshev_filter(
        self,
        operator_matrix: np.ndarray,
        features: np.ndarray,
        coefficients: np.ndarray,
    ) -> np.ndarray:
        coeffs = np.asarray(coefficients, dtype=float)
        k_max = coeffs.size - 1

        t0 = features
        output = coeffs[0] * t0

        if k_max == 0:
            return output

        t1 = operator_matrix @ features
        output = output + coeffs[1] * t1

        for k in range(2, k_max + 1):
            t2 = 2.0 * (operator_matrix @ t1) - t0
            output = output + coeffs[k] * t2
            t0, t1 = t1, t2

        return output


class PolynomialBlockEncodingOperator(BaseQuantumOperatorModel):
    """
    Simulated polynomial operator model inspired by block encoding.

    This class does not claim to implement full QSVT or an exact block-encoded
    hardware routine. Instead, it exposes the operator-theoretic structure:
        g_theta(L) = sum_k c_k L^k
    where L is first normalized into a spectral interval compatible with a
    block-encoding-style interpretation.

    Parameters
    ----------
    coefficients:
        Polynomial coefficients [c_0, c_1, ..., c_K].
    operator_scale:
        Scaling factor alpha such that ||L / alpha|| <= 1 is encouraged.
        If None, a spectral-norm estimate is computed from the matrix.
    renormalize_by_success_probability:
        If True, apply a simple scalar correction inspired by post-selection
        amplitude loss. This is only a lightweight heuristic.
    success_probability_floor:
        Lower bound used in the post-selection-inspired renormalization.
    """

    def __init__(
        self,
        coefficients: list[float] | np.ndarray,
        operator_scale: Optional[float] = None,
        renormalize_by_success_probability: bool = False,
        success_probability_floor: float = 1e-8,
    ) -> None:
        coeffs = np.asarray(coefficients, dtype=float)
        if coeffs.ndim != 1 or coeffs.size == 0:
            raise ValueError("coefficients must be a non-empty 1D array.")

        self.coefficients = coeffs
        self.operator_scale = operator_scale
        self.renormalize_by_success_probability = bool(
            renormalize_by_success_probability
        )
        self.success_probability_floor = float(success_probability_floor)

    def apply(
        self,
        operator_matrix: np.ndarray,
        features: np.ndarray,
    ) -> np.ndarray:
        operator_matrix = np.asarray(operator_matrix, dtype=float)
        features = np.asarray(features, dtype=float)

        if operator_matrix.ndim != 2 or operator_matrix.shape[0] != operator_matrix.shape[1]:
            raise ValueError(
                "operator_matrix must be a square matrix. "
                f"Received shape {operator_matrix.shape}."
            )
        if features.ndim != 2:
            raise ValueError(
                "features must have shape (N, F). "
                f"Received shape {features.shape}."
            )
        if operator_matrix.shape[0] != features.shape[0]:
            raise ValueError(
                "Mismatch between operator size and feature rows. "
                f"Received operator_matrix.shape={operator_matrix.shape} "
                f"and features.shape={features.shape}."
            )

        alpha = self._resolve_operator_scale(operator_matrix)
        normalized_operator = operator_matrix / alpha

        propagated = self._apply_polynomial(normalized_operator, features)

        if self.renormalize_by_success_probability:
            success_proxy = self._estimate_success_probability_proxy(
                normalized_operator,
                features,
            )
            propagated = propagated / np.sqrt(
                max(success_proxy, self.success_probability_floor)
            )

        return propagated

    def _resolve_operator_scale(self, operator_matrix: np.ndarray) -> float:
        if self.operator_scale is not None:
            if self.operator_scale <= 0.0:
                raise ValueError("operator_scale must be strictly positive.")
            return float(self.operator_scale)

        spectral_norm = np.linalg.norm(operator_matrix, ord=2)
        return max(float(spectral_norm), 1.0)

    def _apply_polynomial(
        self,
        normalized_operator: np.ndarray,
        features: np.ndarray,
    ) -> np.ndarray:
        coeffs = self.coefficients
        output = coeffs[0] * features

        if coeffs.size == 1:
            return output

        current_power_applied = features.copy()
        for power in range(1, coeffs.size):
            current_power_applied = normalized_operator @ current_power_applied
            output = output + coeffs[power] * current_power_applied

        return output

    def _estimate_success_probability_proxy(
        self,
        normalized_operator: np.ndarray,
        features: np.ndarray,
    ) -> float:
        test_output = self._apply_polynomial(normalized_operator, features)
        input_norm = np.linalg.norm(features)
        output_norm = np.linalg.norm(test_output)

        if np.isclose(input_norm, 0.0):
            return 1.0

        ratio = output_norm / input_norm
        return float(min(1.0, max(0.0, ratio ** 2)))


class PolynomialBlockEncodingQuantumGraphFilter(BaseQuantumGraphFilter):
    """
    Block-encoding-inspired polynomial graph filter.

    This filter interprets
        g_theta(L_tilde) = sum_k c_k L_tilde^k
    as the target operator and applies it through a simulated polynomial
    operator model. The implementation is designed to be conceptually aligned
    with future block-encoding/QSVT upgrades while remaining usable in the
    current codebase.
    """

    def __init__(
        self,
        coefficients: list[float] | np.ndarray,
        rescale_laplacian: bool = True,
        coefficient_provider: Optional[BaseQuantumCoefficientProvider] = None,
        operator_scale: Optional[float] = None,
        renormalize_by_success_probability: bool = False,
    ) -> None:
        coeffs = np.asarray(coefficients, dtype=float)
        if coeffs.ndim != 1 or coeffs.size == 0:
            raise ValueError("coefficients must be a non-empty 1D array.")

        self.coefficients = coeffs
        self.rescale_laplacian = bool(rescale_laplacian)
        self.coefficient_provider = coefficient_provider
        self.operator_scale = operator_scale
        self.renormalize_by_success_probability = bool(
            renormalize_by_success_probability
        )

    def forward(self, graph_input: GraphInput, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=float)
        if features.ndim != 2:
            raise ValueError(
                "features must have shape (N, F). "
                f"Received shape {features.shape}."
            )
        if features.shape[0] != graph_input.num_nodes:
            raise ValueError(
                "Mismatch between graph size and feature rows. "
                f"Received graph_input.num_nodes={graph_input.num_nodes} "
                f"and features.shape={features.shape}."
            )

        if self.coefficient_provider is None:
            coefficients = self.coefficients
        else:
            coefficients = self.coefficient_provider.get_coefficients(
                graph_input=graph_input,
                features=features,
                num_coefficients=self.coefficients.size,
            )

        if self.rescale_laplacian:
            laplacian = np.asarray(
                graph_input.rescaled_kipf_welling_laplacian(),
                dtype=float,
            )
        else:
            laplacian = np.asarray(
                graph_input.kipf_welling_laplacian(),
                dtype=float,
            )

        operator_model = PolynomialBlockEncodingOperator(
            coefficients=coefficients,
            operator_scale=self.operator_scale,
            renormalize_by_success_probability=(
                self.renormalize_by_success_probability
            ),
        )
        return operator_model.apply(laplacian, features)


class MimicQuantumGraphFilter(BaseQuantumGraphFilter):
    """
    Approximate quantum graph filter based on mixtures of graph-aware
    quantum evolutions.

    The filter builds a surrogate for g_theta(L_tilde) by combining:
    - a residual term proportional to X,
    - one or more evolved terms exp(-i * alpha_r * L_tilde) X,
    - optional real-part decoding and optional subtraction of the residual.

    This class is intended as a practical approximation route for near-term
    experimentation. It is easier to prototype than a true block-encoding
    implementation and still provides a genuinely graph-aware quantum-style
    propagation block.
    """

    def __init__(
        self,
        evolution_times: list[float] | np.ndarray,
        mixture_coefficients: list[float] | np.ndarray,
        residual_coefficient: float = 0.0,
        rescale_laplacian: bool = True,
        take_real_part: bool = True,
        subtract_identity_from_evolution: bool = False,
    ) -> None:
        evolution_times = np.asarray(evolution_times, dtype=float)
        mixture_coefficients = np.asarray(mixture_coefficients, dtype=float)

        if evolution_times.ndim != 1 or evolution_times.size == 0:
            raise ValueError("evolution_times must be a non-empty 1D array.")
        if mixture_coefficients.ndim != 1 or mixture_coefficients.size != evolution_times.size:
            raise ValueError(
                "mixture_coefficients must be a 1D array with the same size "
                "as evolution_times."
            )

        self.evolution_times = evolution_times
        self.mixture_coefficients = mixture_coefficients
        self.residual_coefficient = float(residual_coefficient)
        self.rescale_laplacian = bool(rescale_laplacian)
        self.take_real_part = bool(take_real_part)
        self.subtract_identity_from_evolution = bool(subtract_identity_from_evolution)

    def forward(self, graph_input: GraphInput, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=float)
        if features.ndim != 2:
            raise ValueError(
                "features must have shape (N, F). "
                f"Received shape {features.shape}."
            )
        if features.shape[0] != graph_input.num_nodes:
            raise ValueError(
                "Mismatch between graph size and feature rows. "
                f"Received graph_input.num_nodes={graph_input.num_nodes} "
                f"and features.shape={features.shape}."
            )

        laplacian = self._get_laplacian(graph_input)
        output = self.residual_coefficient * features

        for time_value, mixture_weight in zip(
            self.evolution_times,
            self.mixture_coefficients,
        ):
            evolved = self._apply_time_evolution_to_features(
                laplacian=laplacian,
                features=features,
                alpha=float(time_value),
            )
            if self.subtract_identity_from_evolution:
                evolved = evolved - features
            output = output + float(mixture_weight) * evolved

        return output

    def _get_laplacian(self, graph_input: GraphInput) -> np.ndarray:
        if self.rescale_laplacian:
            return np.asarray(graph_input.rescaled_kipf_welling_laplacian(), dtype=float)
        return np.asarray(graph_input.kipf_welling_laplacian(), dtype=float)

    def _apply_time_evolution_to_features(
        self,
        laplacian: np.ndarray,
        features: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        padded_laplacian = self._pad_square_matrix_to_power_of_two(laplacian)
        unitary = expm(-1j * alpha * padded_laplacian)

        outputs = []
        for feature_index in range(features.shape[1]):
            feature_column = features[:, feature_index]
            outputs.append(
                self._apply_unitary_to_feature(
                    feature_vector=feature_column,
                    unitary=unitary,
                )
            )

        return np.stack(outputs, axis=1)

    def _apply_unitary_to_feature(
        self,
        feature_vector: np.ndarray,
        unitary: np.ndarray,
    ) -> np.ndarray:
        feature_vector = np.asarray(feature_vector, dtype=float)
        original_norm = float(np.linalg.norm(feature_vector))

        if np.isclose(original_norm, 0.0):
            return feature_vector.copy()

        normalized = feature_vector / original_norm
        padded_state = self._pad_vector_to_power_of_two(normalized)
        evolved_state = Statevector(padded_state).evolve(Operator(unitary))

        num_nodes = feature_vector.shape[0]
        decoded = evolved_state.data[:num_nodes]

        if self.take_real_part:
            decoded = np.real(decoded)

        return original_norm * np.asarray(decoded, dtype=float)

    def _pad_vector_to_power_of_two(self, vector: np.ndarray) -> np.ndarray:
        size = vector.shape[0]
        target_size = self._next_power_of_two(size)

        if target_size == size:
            return vector.astype(complex)

        padded = np.zeros(target_size, dtype=complex)
        padded[:size] = vector
        return padded

    def _pad_square_matrix_to_power_of_two(self, matrix: np.ndarray) -> np.ndarray:
        rows, cols = matrix.shape
        if rows != cols:
            raise ValueError(f"matrix must be square. Received shape {matrix.shape}.")

        target_size = self._next_power_of_two(rows)
        if target_size == rows:
            return matrix.astype(complex)

        padded = np.zeros((target_size, target_size), dtype=complex)
        padded[:rows, :cols] = matrix
        return padded

    @staticmethod
    def _next_power_of_two(value: int) -> int:
        if value <= 1:
            return 1
        return 1 << (value - 1).bit_length()


class QuantumGraphLayer:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        quantum_filter: Optional[BaseQuantumGraphFilter] = None,
        activation: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        use_bias: bool = True,
        random_state: Optional[int] = 0,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.quantum_filter = (
            quantum_filter if quantum_filter is not None else ExponentialQuantumGraphFilter()
        )
        self.activation = activation if activation is not None else identity
        self.use_bias = use_bias

        rng = np.random.default_rng(random_state)
        self.weight = 0.1 * rng.standard_normal((in_features, out_features))

        if use_bias:
            self.bias = np.zeros(out_features, dtype=float)
        else:
            self.bias = None

    def forward(
        self,
        graph_input: GraphInput,
        features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if features is None:
            features = graph_input.node_signals

        if features is None:
            raise ValueError(
                "No input features provided. Either pass `features` explicitly "
                "or set `graph_input.node_signals`."
            )

        features = np.asarray(features, dtype=float)
        if features.ndim != 2:
            raise ValueError(
                "features must have shape (N, F_in). "
                f"Received shape {features.shape}."
            )

        if features.shape[1] != self.in_features:
            raise ValueError(
                "features.shape[1] must match in_features. "
                f"Received features.shape[1]={features.shape[1]} "
                f"and in_features={self.in_features}."
            )

        propagated = self.quantum_filter.forward(graph_input, features)
        output = propagated @ self.weight

        if self.bias is not None:
            output = output + self.bias

        return self.activation(output)


class SingleLayerQuantumGraphNetwork:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        quantum_filter: Optional[BaseQuantumGraphFilter] = None,
        activation: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        use_bias: bool = True,
        random_state: Optional[int] = 0,
    ) -> None:
        self.layer = QuantumGraphLayer(
            in_features=in_features,
            out_features=out_features,
            quantum_filter=quantum_filter,
            activation=activation,
            use_bias=use_bias,
            random_state=random_state,
        )

    def forward(
        self,
        graph_input: GraphInput,
        features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self.layer.forward(graph_input, features)


class TwoLayerQuantumGraphNetwork:
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

    def forward(
        self,
        graph_input: GraphInput,
        features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        hidden = self.layer1.forward(graph_input, features)
        return self.layer2.forward(graph_input, hidden)
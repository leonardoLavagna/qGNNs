from __future__ import annotations

# -----------------------------------------------------------------------------
# qcheb_gnn.py
#
# Quantum/Chebyshev graph filters compatible with the existing GraphInput and
# qgtheta/shallow_kipf_welling_gnn APIs.
#
# Main responsibilities:
# - provide a Chebyshev-approximated Kipf-Welling spectral filter
# - support a direct non-diagonalizing recurrence backend as the default path
# - support an exact spectral backend for small-graph validation/debugging
# - separate coefficient generation, polynomial evaluation, operator selection,
#   and execution backend concerns so that future diagonalization-based methods
#   (for example SQD-like spectral pipelines) can be added with minimal changes
# - provide one-layer and two-layer network wrappers analogous to qgtheta.py
#
# Design note:
# The default path in this module does NOT require diagonalization. The Chebyshev
# expansion is applied directly through the recurrence
#
#     T_0(L)X = X
#     T_1(L)X = LX
#     T_k(L)X = 2 L T_{k-1}(L)X - T_{k-2}(L)X.
#
# To prepare for future diagonalization approaches, the module also defines an
# explicit spectral-backend interface. At present, an exact dense-eigendecomposition
# backend is included for debugging/reference on small graphs. A future SQD-based
# backend can implement the same interface and plug into the filter without
# refactoring the layer/network APIs.
#
# © Leonardo Lavagna 2026
# -----------------------------------------------------------------------------


from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from classical_gnns.shallow_kipf_welling_gnn import identity, relu
from utils.graphs import GraphInput
from quantum_gnns.qgtheta import BaseQuantumCoefficientProvider,QiskitVariationalCoefficientProvider


Activation = Callable[[np.ndarray], np.ndarray]



# -----------------------------------------------------------------------------
# Polynomial providers
# -----------------------------------------------------------------------------
class BaseQChebPolynomialProvider(ABC):
    """Interface for providers that return Chebyshev basis values T_k(x)."""

    @abstractmethod
    def evaluate_basis(self, x: np.ndarray, order: int) -> np.ndarray:
        """Return basis matrix B with B[i, k] = T_k(x_i)."""
        raise NotImplementedError


class ClassicalChebyshevPolynomialProvider(BaseQChebPolynomialProvider):
    """Reference provider using the standard Chebyshev recurrence."""

    def evaluate_basis(self, x: np.ndarray, order: int) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if order < 0:
            raise ValueError("order must be non-negative.")

        basis = np.zeros((x.size, order + 1), dtype=float)
        basis[:, 0] = 1.0
        if order >= 1:
            basis[:, 1] = x
        for k in range(2, order + 1):
            basis[:, k] = 2.0 * x * basis[:, k - 1] - basis[:, k - 2]
        return basis


class QiskitQuantumChebyshevPolynomialProvider(BaseQChebPolynomialProvider):
    """
    Qiskit-native Chebyshev-basis provider.

    The provider uses
        T_k(x) = cos(k arccos(x)),  x in [-1, 1]
    and evaluates cos(k theta) from a one-qubit circuit. For a state prepared as
        RY(phi) |0>,
    the expectation value of Z is cos(phi). Therefore, with phi = k * arccos(x),
    the Z expectation equals T_k(x).
    """

    def __init__(self, clip_input: bool = True) -> None:
        self.clip_input = bool(clip_input)

    def evaluate_basis(self, x: np.ndarray, order: int) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if order < 0:
            raise ValueError("order must be non-negative.")

        if self.clip_input:
            x = np.clip(x, -1.0, 1.0)
        elif np.any(np.abs(x) > 1.0 + 1e-12):
            raise ValueError(
                "All inputs must lie in [-1, 1] for Chebyshev basis evaluation."
            )

        basis = np.zeros((x.size, order + 1), dtype=float)
        basis[:, 0] = 1.0
        for i, value in enumerate(x):
            theta = float(np.arccos(np.clip(value, -1.0, 1.0)))
            for k in range(1, order + 1):
                basis[i, k] = self._chebyshev_value_from_circuit(theta=theta, degree=k)
        return basis

    @staticmethod
    def _chebyshev_value_from_circuit(theta: float, degree: int) -> float:
        qc = QuantumCircuit(1)
        qc.ry(float(degree) * theta, 0)
        state = Statevector.from_instruction(qc)
        value = state.expectation_value(SparsePauliOp("Z"))
        return float(np.real(value))


# -----------------------------------------------------------------------------
# Operator selection / future spectral context
# -----------------------------------------------------------------------------
@dataclass
class OperatorContext:
    """
    Container for the graph operator used by the Chebyshev filter.

    This context is intentionally explicit so that future spectral backends can
    attach additional metadata without changing the external API.
    """

    operator_matrix: np.ndarray
    operator_name: str
    is_rescaled_to_minus_one_one: bool


@dataclass
class SpectralData:
    """
    Spectral representation of the chosen graph operator.

    For now this stores a full orthonormal eigenbasis. In a future SQD-style
    implementation, the same dataclass can be populated approximately from
    quantum samples, possibly with only a selected subspace or a compressed set
    of modes.
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    is_exact: bool = True
    metadata: Optional[dict] = None


class BaseGraphOperatorProvider(ABC):
    """Interface for selecting the graph operator used by the filter."""

    @abstractmethod
    def build(self, graph_input: GraphInput) -> OperatorContext:
        raise NotImplementedError


class KipfWellingLaplacianOperatorProvider(BaseGraphOperatorProvider):
    """
    Operator provider for the Kipf-Welling Laplacian-like operator.

    Parameters:
        use_rescaled_laplacian:
            If True, use the spectrally rescaled version of
            L_tilde = I - A_hat, which is the most natural choice for Chebyshev
            expansions because its spectrum is approximately mapped to [-1, 1].
    """

    def __init__(self, use_rescaled_laplacian: bool = True) -> None:
        self.use_rescaled_laplacian = bool(use_rescaled_laplacian)

    def build(self, graph_input: GraphInput) -> OperatorContext:
        if self.use_rescaled_laplacian:
            matrix = np.asarray(
                graph_input.rescaled_kipf_welling_laplacian(),
                dtype=float,
            )
            return OperatorContext(
                operator_matrix=matrix,
                operator_name="rescaled_kipf_welling_laplacian",
                is_rescaled_to_minus_one_one=True,
            )

        matrix = np.asarray(graph_input.kipf_welling_laplacian(), dtype=float)
        return OperatorContext(
            operator_matrix=matrix,
            operator_name="kipf_welling_laplacian",
            is_rescaled_to_minus_one_one=False,
        )


class BaseSpectralBackend(ABC):
    """
    Interface for producing spectral information about an operator.

    This is the main extension point for future diagonalization-based methods,
    including approximate or sample-based spectral workflows.
    """

    @abstractmethod
    def compute_spectral_data(
        self,
        operator_context: OperatorContext,
        graph_input: GraphInput,
        features: np.ndarray,
    ) -> SpectralData:
        raise NotImplementedError


class ExactEigendecompositionBackend(BaseSpectralBackend):
    """Reference dense eigendecomposition backend for small graphs."""

    def compute_spectral_data(
        self,
        operator_context: OperatorContext,
        graph_input: GraphInput,
        features: np.ndarray,
    ) -> SpectralData:
        matrix = np.asarray(operator_context.operator_matrix, dtype=float)
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        return SpectralData(
            eigenvalues=np.asarray(eigenvalues, dtype=float),
            eigenvectors=np.asarray(eigenvectors, dtype=float),
            is_exact=True,
            metadata={"backend": "numpy.linalg.eigh"},
        )


class PlaceholderSQDSpectralBackend(BaseSpectralBackend):
    """
    Placeholder interface for future SQD-like spectral estimation.

    This class is intentionally not implemented yet. Its purpose is to pin down
    the interface that a future sample-based diagonalization backend should
    satisfy, so the rest of the module does not need structural refactoring.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def compute_spectral_data(
        self,
        operator_context: OperatorContext,
        graph_input: GraphInput,
        features: np.ndarray,
    ) -> SpectralData:
        raise NotImplementedError(
            "PlaceholderSQDSpectralBackend is a design placeholder only. "
            "Later, replace this with a backend that builds an SQD-compatible "
            "subspace/eigensystem approximation from quantum samples."
        )


# -----------------------------------------------------------------------------
# Execution backends
# -----------------------------------------------------------------------------
class BaseChebyshevExecutionBackend(ABC):
    """Interface for applying the Chebyshev filter to feature matrices."""

    @abstractmethod
    def apply(
        self,
        operator_context: OperatorContext,
        features: np.ndarray,
        theta: np.ndarray,
        polynomial_provider: BaseQChebPolynomialProvider,
        graph_input: GraphInput,
    ) -> np.ndarray:
        raise NotImplementedError


class DirectChebyshevRecurrenceBackend(BaseChebyshevExecutionBackend):
    """
    Default backend that applies the filter directly without diagonalization.

    This backend is the recommended starting point because it is stable,
    efficient for moderate graph sizes, and keeps the implementation close to
    the original Chebyshev-GNN recurrence.
    """

    def apply(
        self,
        operator_context: OperatorContext,
        features: np.ndarray,
        theta: np.ndarray,
        polynomial_provider: BaseQChebPolynomialProvider,
        graph_input: GraphInput,
    ) -> np.ndarray:
        del polynomial_provider, graph_input

        operator_matrix = np.asarray(operator_context.operator_matrix, dtype=float)
        if theta.ndim != 1:
            raise ValueError("theta must be a 1D array.")

        order = theta.size - 1
        if order < 0:
            raise ValueError("theta must contain at least one coefficient.")

        if order == 0:
            return float(theta[0]) * features

        t0 = features.copy()
        output = float(theta[0]) * t0

        t1 = operator_matrix @ features
        output = output + float(theta[1]) * t1

        for k in range(2, order + 1):
            t2 = 2.0 * operator_matrix @ t1 - t0
            output = output + float(theta[k]) * t2
            t0, t1 = t1, t2

        return output


class SpectralChebyshevExecutionBackend(BaseChebyshevExecutionBackend):
    """
    Spectral backend that separates basis evaluation from spectral estimation.

    This backend is mostly useful as a reference implementation and as the main
    architectural hook for later diagonalization-based approaches.
    """

    def __init__(
        self,
        spectral_backend: Optional[BaseSpectralBackend] = None,
    ) -> None:
        self.spectral_backend = (
            spectral_backend
            if spectral_backend is not None
            else ExactEigendecompositionBackend()
        )

    def apply(
        self,
        operator_context: OperatorContext,
        features: np.ndarray,
        theta: np.ndarray,
        polynomial_provider: BaseQChebPolynomialProvider,
        graph_input: GraphInput,
    ) -> np.ndarray:
        spectral_data = self.spectral_backend.compute_spectral_data(
            operator_context=operator_context,
            graph_input=graph_input,
            features=features,
        )

        basis = polynomial_provider.evaluate_basis(
            spectral_data.eigenvalues,
            order=theta.size - 1,
        )
        spectral_response = basis @ theta

        projected = spectral_data.eigenvectors.T @ features
        filtered_projected = spectral_response[:, None] * projected
        return spectral_data.eigenvectors @ filtered_projected


# -----------------------------------------------------------------------------
# Main filter
# -----------------------------------------------------------------------------
class ChebyshevKipfWellingGraphFilter:
    """
    Chebyshev approximation to the spectral Kipf-Welling filter.

    The filter applies
        g_theta(L) X = sum_{k=0}^K theta_k T_k(L_used) X
    where the default operator is the rescaled Kipf-Welling Laplacian.

    Architecture:
    - operator_provider decides which graph operator is used
    - coefficient_provider optionally generates theta_k dynamically
    - execution_backend decides how the polynomial is applied
    - polynomial_provider evaluates T_k(x) when a spectral backend needs it

    This separation makes it straightforward to later replace the exact spectral
    backend with an SQD-style approximate diagonalization backend, while keeping
    the public forward(graph_input, features) API unchanged.
    """

    def __init__(
        self,
        order: int = 2,
        theta: Optional[np.ndarray] = None,
        coefficient_provider: Optional[BaseQuantumCoefficientProvider] = None,
        polynomial_provider: Optional[BaseQChebPolynomialProvider] = None,
        operator_provider: Optional[BaseGraphOperatorProvider] = None,
        execution_backend: Optional[BaseChebyshevExecutionBackend] = None,
        validate_symmetry: bool = True,
    ) -> None:
        if order < 0:
            raise ValueError("order must be non-negative.")

        self.order = int(order)
        self.coefficient_provider = coefficient_provider
        self.polynomial_provider = (
            polynomial_provider
            if polynomial_provider is not None
            else ClassicalChebyshevPolynomialProvider()
        )
        self.operator_provider = (
            operator_provider
            if operator_provider is not None
            else KipfWellingLaplacianOperatorProvider(use_rescaled_laplacian=True)
        )
        self.execution_backend = (
            execution_backend
            if execution_backend is not None
            else DirectChebyshevRecurrenceBackend()
        )
        self.validate_symmetry = bool(validate_symmetry)

        if theta is None:
            theta_array = np.zeros(self.order + 1, dtype=float)
            theta_array[0] = 1.0
        else:
            theta_array = np.asarray(theta, dtype=float).reshape(-1)
            if theta_array.size != self.order + 1:
                raise ValueError(
                    f"theta must have length {self.order + 1}. "
                    f"Received {theta_array.size}."
                )
        self.theta = theta_array

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

        operator_context = self.operator_provider.build(graph_input)
        self._validate_operator(operator_context.operator_matrix)
        theta = self._resolve_theta(graph_input, features)

        return self.execution_backend.apply(
            operator_context=operator_context,
            features=features,
            theta=theta,
            polynomial_provider=self.polynomial_provider,
            graph_input=graph_input,
        )

    def _validate_operator(self, matrix: np.ndarray) -> None:
        if not self.validate_symmetry:
            return

        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("The chosen operator must be square.")
        if not np.allclose(matrix, matrix.T, atol=1e-10):
            raise ValueError("The chosen operator must be symmetric/Hermitian.")

    def _resolve_theta(self, graph_input: GraphInput, features: np.ndarray) -> np.ndarray:
        if self.coefficient_provider is None:
            return self.theta.copy()

        theta = np.asarray(
            self.coefficient_provider.get_coefficients(
                graph_input=graph_input,
                features=features,
                num_coefficients=self.order + 1,
            ),
            dtype=float,
        ).reshape(-1)

        if theta.size != self.order + 1:
            raise ValueError(
                "Coefficient provider returned an unexpected number of coefficients. "
                f"Expected {self.order + 1}, received {theta.size}."
            )
        return theta


# -----------------------------------------------------------------------------
# Layers and networks
# -----------------------------------------------------------------------------
@dataclass
class QChebGraphLayer:
    """Layer analogous to QuantumGraphLayer but using a Chebyshev graph filter."""

    weight: np.ndarray
    cheb_filter: ChebyshevKipfWellingGraphFilter
    bias: Optional[np.ndarray] = None
    activation: Activation = identity

    def forward(
        self,
        graph_input: GraphInput,
        features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        x = self._resolve_features(graph_input, features)
        propagated = self.cheb_filter.forward(graph_input, x)
        z = propagated @ self.weight

        if self.bias is not None:
            z = z + self.bias

        return self.activation(z)

    def _resolve_features(
        self,
        graph_input: GraphInput,
        features: Optional[np.ndarray],
    ) -> np.ndarray:
        if features is None:
            if graph_input.node_signals is None:
                raise ValueError(
                    "No input features provided. Pass `features` explicitly "
                    "or set `graph_input.node_signals`."
                )
            x = graph_input.node_signals
        else:
            x = np.asarray(features, dtype=float)

        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.shape[0] != graph_input.num_nodes:
            raise ValueError("Feature matrix row count must match the number of graph nodes.")
        if x.shape[1] != self.weight.shape[0]:
            raise ValueError("Feature dimension and weight input dimension do not match.")

        return x


class SingleLayerQChebNetwork:
    """Single-layer Chebyshev graph network."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        cheb_filter: Optional[ChebyshevKipfWellingGraphFilter] = None,
        activation: Optional[Activation] = None,
        use_bias: bool = True,
        random_state: Optional[int] = 0,
    ) -> None:
        rng = np.random.default_rng(random_state)
        weight = 0.1 * rng.standard_normal((in_features, out_features))
        bias = np.zeros(out_features, dtype=float) if use_bias else None
        self.layer = QChebGraphLayer(
            weight=weight,
            cheb_filter=cheb_filter if cheb_filter is not None else ChebyshevKipfWellingGraphFilter(),
            bias=bias,
            activation=activation if activation is not None else identity,
        )

    def forward(
        self,
        graph_input: GraphInput,
        features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self.layer.forward(graph_input, features)


class TwoLayerQChebNetwork:
    """Two-layer Chebyshev graph network, API-compatible with qgtheta wrappers."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        first_filter: Optional[ChebyshevKipfWellingGraphFilter] = None,
        second_filter: Optional[ChebyshevKipfWellingGraphFilter] = None,
        hidden_activation: Optional[Activation] = relu,
        output_activation: Optional[Activation] = None,
        random_state: Optional[int] = 0,
    ) -> None:
        rng1 = np.random.default_rng(random_state)
        rng2 = np.random.default_rng(None if random_state is None else random_state + 1)

        self.layer1 = QChebGraphLayer(
            weight=0.1 * rng1.standard_normal((in_features, hidden_features)),
            cheb_filter=first_filter if first_filter is not None else ChebyshevKipfWellingGraphFilter(),
            bias=np.zeros(hidden_features, dtype=float),
            activation=hidden_activation if hidden_activation is not None else identity,
        )
        self.layer2 = QChebGraphLayer(
            weight=0.1 * rng2.standard_normal((hidden_features, out_features)),
            cheb_filter=second_filter if second_filter is not None else ChebyshevKipfWellingGraphFilter(),
            bias=np.zeros(out_features, dtype=float),
            activation=output_activation if output_activation is not None else identity,
        )

    def forward(
        self,
        graph_input: GraphInput,
        features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        hidden = self.layer1.forward(graph_input, features)
        return self.layer2.forward(graph_input, hidden)


# -----------------------------------------------------------------------------
# Convenience constructors
# -----------------------------------------------------------------------------
def build_direct_cheb_filter(
    order: int,
    theta: np.ndarray,
    use_rescaled_laplacian: bool = True,
) -> ChebyshevKipfWellingGraphFilter:
    """
    Convenience constructor for the recommended starting point.

    This path uses the direct recurrence backend and does not require any
    diagonalization.
    """
    return ChebyshevKipfWellingGraphFilter(
        order=order,
        theta=theta,
        coefficient_provider=None,
        polynomial_provider=ClassicalChebyshevPolynomialProvider(),
        operator_provider=KipfWellingLaplacianOperatorProvider(
            use_rescaled_laplacian=use_rescaled_laplacian,
        ),
        execution_backend=DirectChebyshevRecurrenceBackend(),
    )



def build_qct_basis_direct_cheb_filter(
    order: int,
    theta: np.ndarray,
    use_rescaled_laplacian: bool = True,
) -> ChebyshevKipfWellingGraphFilter:
    """
    Hybrid constructor that keeps the main filter direct but prepares the
    qiskit-based polynomial provider for later spectral experiments.

    Note: in the current direct backend the polynomial provider is not used,
    because no spectral basis evaluation is needed. This constructor is included
    mainly for API symmetry with the spectral route and for future experiments.
    """
    return ChebyshevKipfWellingGraphFilter(
        order=order,
        theta=theta,
        coefficient_provider=None,
        polynomial_provider=QiskitQuantumChebyshevPolynomialProvider(),
        operator_provider=KipfWellingLaplacianOperatorProvider(
            use_rescaled_laplacian=use_rescaled_laplacian,
        ),
        execution_backend=DirectChebyshevRecurrenceBackend(),
    )



def build_exact_spectral_qct_cheb_filter(
    order: int,
    theta: np.ndarray,
    use_rescaled_laplacian: bool = True,
) -> ChebyshevKipfWellingGraphFilter:
    """
    Reference constructor for small-graph spectral debugging.

    This path uses an exact eigendecomposition backend together with the Qiskit
    polynomial provider.
    """
    return ChebyshevKipfWellingGraphFilter(
        order=order,
        theta=theta,
        coefficient_provider=None,
        polynomial_provider=QiskitQuantumChebyshevPolynomialProvider(),
        operator_provider=KipfWellingLaplacianOperatorProvider(
            use_rescaled_laplacian=use_rescaled_laplacian,
        ),
        execution_backend=SpectralChebyshevExecutionBackend(
            spectral_backend=ExactEigendecompositionBackend(),
        ),
    )



def build_fully_quantum_exact_spectral_qct_cheb_filter(
    order: int,
    coefficient_provider: Optional[BaseQuantumCoefficientProvider] = None,
    use_rescaled_laplacian: bool = True,
) -> ChebyshevKipfWellingGraphFilter:
    """
    Constructor for the fully hybrid route:
    - spectral coefficients theta_k from a QiskitVariationalCoefficientProvider
    - Chebyshev basis values from the Qiskit polynomial provider
    - exact spectral backend for current small-graph validation
    """
    provider = (
        coefficient_provider
        if coefficient_provider is not None
        else QiskitVariationalCoefficientProvider(
            num_qubits=2,
            num_layers=1,
            output_scale=0.5,
            normalize_coefficients=False,
        )
    )

    return ChebyshevKipfWellingGraphFilter(
        order=order,
        theta=None,
        coefficient_provider=provider,
        polynomial_provider=QiskitQuantumChebyshevPolynomialProvider(),
        operator_provider=KipfWellingLaplacianOperatorProvider(
            use_rescaled_laplacian=use_rescaled_laplacian,
        ),
        execution_backend=SpectralChebyshevExecutionBackend(
            spectral_backend=ExactEigendecompositionBackend(),
        ),
    )

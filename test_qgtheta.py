# -----------------------------------------------------------------------------
# test_qgtheta.py
#
# Extended test and demo script for quantum_gnns.qgtheta.
#
# This script:
# - builds a small demo graph with node and edge signals,
# - computes a classical Kipf-Welling baseline,
# - compares multiple quantum / hybrid graph filters against that baseline,
# - prints hidden/output activations and Frobenius-norm differences,
# - visualizes hidden-layer channels across models.
#
# Models tested:
# - Classical two-layer baseline
# - First-order spectral filter (static)
# - First-order spectral filter (quantum-controlled coefficients)
# - Chebyshev spectral filter (static)
# - Chebyshev spectral filter (quantum-controlled coefficients)
# - Exponential quantum filter
# - Polynomial block-encoding-inspired filter (static)
# - Polynomial block-encoding-inspired filter (quantum-controlled coefficients)
# - Mimic / approximation quantum filter
#
# Expected project structure:
# - utils/graphs.py
# - classical_gnns/shallow_kipf_welling_gnn.py
# - quantum_gnns/qgtheta.py
#
# © Leonardo Lavagna 2026
# -----------------------------------------------------------------------------
from __future__ import annotations

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from quantum_gnns.qgtheta import *
from utils.graphs import build_graph_input


# -----------------------------------------------------------------------------
# Classical baseline
# -----------------------------------------------------------------------------
class ClassicalGraphLayer:
    """
    Simple classical Kipf-Welling-style graph layer:

        H_out = activation(A_hat @ H_in @ W + b)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation=identity,
        use_bias: bool = False,
        random_state: int | None = None,
    ) -> None:
        rng = np.random.default_rng(random_state)
        self.weight = rng.normal(loc=0.0, scale=0.5, size=(in_features, out_features))
        self.bias = np.zeros(out_features, dtype=float) if use_bias else None
        self.activation = activation

    def forward(self, graph_input, features: np.ndarray | None = None) -> np.ndarray:
        if features is None:
            features = np.asarray(graph_input.node_signals, dtype=float)
        else:
            features = np.asarray(features, dtype=float)

        a_hat = np.asarray(graph_input.kipf_welling_adjacency(), dtype=float)
        output = a_hat @ features @ self.weight
        if self.bias is not None:
            output = output + self.bias
        return self.activation(output)


class ClassicalTwoLayerGraphNetwork:
    """Two-layer classical reference GNN."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        hidden_activation=relu,
        output_activation=identity,
        random_state: int | None = None,
    ) -> None:
        self.layer1 = ClassicalGraphLayer(
            in_features=in_features,
            out_features=hidden_features,
            activation=hidden_activation,
            use_bias=False,
            random_state=random_state,
        )
        self.layer2 = ClassicalGraphLayer(
            in_features=hidden_features,
            out_features=out_features,
            activation=output_activation,
            use_bias=False,
            random_state=None if random_state is None else random_state + 1,
        )

    def forward(self, graph_input) -> tuple[np.ndarray, np.ndarray]:
        h1 = self.layer1.forward(graph_input)
        h2 = self.layer2.forward(graph_input, features=h1)
        return h1, h2


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def build_demo_graph_input():
    """Create a small demo graph and signals."""
    G = nx.path_graph(5)

    node_signals = [
        [1.0, 0.0],
        [0.5, 0.2],
        [0.0, 1.0],
        [0.2, 0.5],
        [1.0, 1.0],
    ]

    edge_signals = {
        (0, 1): [1.0],
        (1, 2): [0.7],
        (2, 3): [1.2],
        (3, 4): [0.9],
    }

    graph_input = build_graph_input(
        G,
        node_signals=node_signals,
        edge_signals=edge_signals,
    )
    return G, graph_input


def print_block(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def print_model_summary(
    name: str,
    h1_model: np.ndarray,
    h2_model: np.ndarray,
    h1_classical: np.ndarray,
    h2_classical: np.ndarray,
) -> None:
    """Print model outputs and differences against the classical baseline."""
    print_block(name)
    print("H1:\n", h1_model)
    print("H1 shape:", h1_model.shape)
    print("H2:\n", h2_model)
    print("H2 shape:", h2_model.shape)

    diff_h1 = h1_model - h1_classical
    diff_h2 = h2_model - h2_classical

    print("\nDifference H1 (model - classical):\n", diff_h1)
    print("Difference H2 (model - classical):\n", diff_h2)
    print("||H1_model - H1_classical||_F =", np.linalg.norm(diff_h1))
    print("||H2_model - H2_classical||_F =", np.linalg.norm(diff_h2))


def print_generated_coefficients(
    title: str,
    provider: QiskitVariationalCoefficientProvider,
    graph_input,
    features: np.ndarray,
    num_coefficients: int,
) -> None:
    coeffs = provider.get_coefficients(
        graph_input=graph_input,
        features=features,
        num_coefficients=num_coefficients,
    )
    print(f"{title}: {coeffs}")


def build_two_layer_quantum_model(
    in_features: int,
    hidden_features: int,
    out_features: int,
    first_filter,
    second_filter,
    hidden_activation,
    output_activation,
    random_state: int | None,
    W1: np.ndarray,
    W2: np.ndarray,
):
    """
    Build a TwoLayerQuantumGraphNetwork and overwrite the trainable linear
    weights with the classical baseline weights for fair comparison.
    """
    model = TwoLayerQuantumGraphNetwork(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        first_filter=first_filter,
        second_filter=second_filter,
        hidden_activation=hidden_activation,
        output_activation=output_activation,
        random_state=random_state,
    )

    model.layer1.weight = W1.copy()
    model.layer2.weight = W2.copy()
    model.layer1.bias = None
    model.layer2.bias = None
    return model


def plot_all_hidden_comparisons(
    G,
    graph_input,
    model_hidden_dict: dict[str, np.ndarray],
    baseline_key: str = "Classical",
    layout_seed: int = 42,
) -> None:
    """
    Plot all hidden-layer comparisons in one figure.

    Parameters
    ----------
    G:
        NetworkX graph.
    graph_input:
        GraphInput object with edge signals.
    model_hidden_dict:
        Dictionary mapping model names to hidden-layer matrices of shape
        (num_nodes, hidden_features).
    baseline_key:
        Name of the baseline model used for difference rows.
    layout_seed:
        Seed for graph layout reproducibility.
    """
    pos = nx.spring_layout(G, seed=layout_seed)

    edge_values = []
    for u, v in G.edges():
        key = (u, v) if (u, v) in graph_input.edge_signals else (v, u)
        edge_signal = graph_input.edge_signals[key]
        if isinstance(edge_signal, (list, tuple, np.ndarray)):
            edge_values.append(float(edge_signal[0]))
        else:
            edge_values.append(float(edge_signal))

    baseline = model_hidden_dict[baseline_key]

    model_arrays = list(model_hidden_dict.values())
    all_hidden_vals = np.concatenate([arr.reshape(-1) for arr in model_arrays])
    vmin_hidden = float(np.min(all_hidden_vals))
    vmax_hidden = float(np.max(all_hidden_vals))

    diff_dict = {
        f"{name} - {baseline_key}": arr - baseline
        for name, arr in model_hidden_dict.items()
        if name != baseline_key
    }

    all_diff_vals = np.concatenate([arr.reshape(-1) for arr in diff_dict.values()])
    dmax_global = float(np.max(np.abs(all_diff_vals))) if all_diff_vals.size > 0 else 1.0
    if dmax_global < 1e-12:
        dmax_global = 1.0

    num_hidden_channels = baseline.shape[1]

    row_data: list[tuple[str, np.ndarray, str]] = []
    for name, arr in model_hidden_dict.items():
        row_data.append((name, arr, "model"))
    for name, arr in diff_dict.items():
        row_data.append((name, arr, "diff"))

    nrows = len(row_data)
    fig, axes = plt.subplots(
        nrows,
        num_hidden_channels,
        figsize=(5 * num_hidden_channels, 3.2 * nrows),
    )

    if nrows == 1 and num_hidden_channels == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif num_hidden_channels == 1:
        axes = np.array(axes).reshape(nrows, 1)

    for row_idx, (row_title, row_matrix, row_kind) in enumerate(row_data):
        for ch in range(num_hidden_channels):
            values = row_matrix[:, ch]

            if row_kind == "model":
                nx.draw(
                    G,
                    pos,
                    with_labels=True,
                    node_color=values,
                    edge_color=edge_values,
                    cmap="viridis",
                    edge_cmap=plt.cm.plasma,
                    vmin=vmin_hidden,
                    vmax=vmax_hidden,
                    node_size=700,
                    ax=axes[row_idx, ch],
                )
            else:
                nx.draw(
                    G,
                    pos,
                    with_labels=True,
                    node_color=values,
                    edge_color=edge_values,
                    cmap="coolwarm",
                    edge_cmap=plt.cm.plasma,
                    vmin=-dmax_global,
                    vmax=dmax_global,
                    node_size=700,
                    ax=axes[row_idx, ch],
                )

            axes[row_idx, ch].set_title(f"{row_title} ch {ch}")

    fig.suptitle("Hidden-layer comparisons across all models", fontsize=16)
    plt.tight_layout()
    plt.show()


def print_cross_summary(
    baseline_name: str,
    baseline_h1: np.ndarray,
    baseline_h2: np.ndarray,
    model_outputs: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """Print Frobenius-norm differences of all models against the baseline."""
    print_block("Cross-model Frobenius-norm summary")

    for name, (h1, h2) in model_outputs.items():
        if name == baseline_name:
            continue
        print(f"||H1_{name} - H1_{baseline_name}||_F = {np.linalg.norm(h1 - baseline_h1)}")
        print(f"||H2_{name} - H2_{baseline_name}||_F = {np.linalg.norm(h2 - baseline_h2)}")

    print_block("Cross-model pairwise summaries (selected)")
    selected_pairs = [
        ("First-order quantum", "First-order static"),
        ("Chebyshev quantum", "Chebyshev static"),
        ("Block polynomial quantum", "Block polynomial static"),
        ("Mimic quantum", "Block polynomial static"),
    ]

    for a, b in selected_pairs:
        if a in model_outputs and b in model_outputs:
            h1_a, h2_a = model_outputs[a]
            h1_b, h2_b = model_outputs[b]
            print(f"||H1_{a} - H1_{b}||_F = {np.linalg.norm(h1_a - h1_b)}")
            print(f"||H2_{a} - H2_{b}||_F = {np.linalg.norm(h2_a - h2_b)}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    np.set_printoptions(precision=5, suppress=True)

    # -------------------------------------------------------------------------
    # Build data
    # -------------------------------------------------------------------------
    G, graph_input = build_demo_graph_input()

    print_block("Graph input")
    print("Node order:", graph_input.node_order)
    print("Node signals:\n", graph_input.node_signals)
    print("Edge signals:\n", graph_input.edge_signals)
    print("Kipf-Welling adjacency A_hat:\n", graph_input.kipf_welling_adjacency())
    print("Kipf-Welling Laplacian:\n", graph_input.kipf_welling_laplacian())
    print("Rescaled Kipf-Welling Laplacian:\n", graph_input.rescaled_kipf_welling_laplacian())

    # -------------------------------------------------------------------------
    # Shared setup
    # -------------------------------------------------------------------------
    in_features = 2
    hidden_features = 3
    out_features = 1
    random_state = 0

    model_outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # -------------------------------------------------------------------------
    # Classical baseline
    # -------------------------------------------------------------------------
    classical_model = ClassicalTwoLayerGraphNetwork(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        hidden_activation=relu,
        output_activation=identity,
        random_state=random_state,
    )

    W1 = classical_model.layer1.weight.copy()
    W2 = classical_model.layer2.weight.copy()

    H1_classical, H2_classical = classical_model.forward(graph_input)
    model_outputs["Classical"] = (H1_classical, H2_classical)

    print_block("Classical baseline")
    print("W1:\n", W1)
    print("W2:\n", W2)
    print("H1_classical:\n", H1_classical)
    print("H2_classical:\n", H2_classical)

    # -------------------------------------------------------------------------
    # Static first-order filter
    # -------------------------------------------------------------------------
    first_static_filter_1 = FirstOrderQuantumGraphFilter(theta_0=0.0, theta_1=1.0)
    first_static_filter_2 = FirstOrderQuantumGraphFilter(theta_0=0.0, theta_1=1.0)

    first_static_model = build_two_layer_quantum_model(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        first_filter=first_static_filter_1,
        second_filter=first_static_filter_2,
        hidden_activation=relu,
        output_activation=identity,
        random_state=random_state,
        W1=W1,
        W2=W2,
    )

    H1_first_static = first_static_model.layer1.forward(graph_input)
    H2_first_static = first_static_model.layer2.forward(graph_input, features=H1_first_static)
    model_outputs["First-order static"] = (H1_first_static, H2_first_static)

    print_model_summary(
        "Static first-order spectral filter",
        H1_first_static,
        H2_first_static,
        H1_classical,
        H2_classical,
    )

    # -------------------------------------------------------------------------
    # Quantum-controlled first-order filter
    # -------------------------------------------------------------------------
    first_provider = QiskitVariationalCoefficientProvider(
        num_qubits=2,
        num_layers=2,
        input_scale=0.5,
        output_scale=1.0,
        output_bias=np.array([0.0, 1.0]),
    )

    print_generated_coefficients(
        title="Quantum-generated first-order coefficients on input features",
        provider=first_provider,
        graph_input=graph_input,
        features=np.asarray(graph_input.node_signals, dtype=float),
        num_coefficients=2,
    )

    first_quantum_filter_1 = FirstOrderQuantumGraphFilter(
        theta_0=0.0,
        theta_1=1.0,
        coefficient_provider=first_provider,
    )
    first_quantum_filter_2 = FirstOrderQuantumGraphFilter(
        theta_0=0.0,
        theta_1=1.0,
        coefficient_provider=first_provider,
    )

    first_quantum_model = build_two_layer_quantum_model(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        first_filter=first_quantum_filter_1,
        second_filter=first_quantum_filter_2,
        hidden_activation=relu,
        output_activation=identity,
        random_state=random_state,
        W1=W1,
        W2=W2,
    )

    H1_first_quantum = first_quantum_model.layer1.forward(graph_input)

    print_generated_coefficients(
        title="Quantum-generated first-order coefficients after layer 1",
        provider=first_provider,
        graph_input=graph_input,
        features=H1_first_quantum,
        num_coefficients=2,
    )

    H2_first_quantum = first_quantum_model.layer2.forward(
        graph_input,
        features=H1_first_quantum,
    )
    model_outputs["First-order quantum"] = (H1_first_quantum, H2_first_quantum)

    print_model_summary(
        "Quantum-controlled first-order spectral filter",
        H1_first_quantum,
        H2_first_quantum,
        H1_classical,
        H2_classical,
    )

    # -------------------------------------------------------------------------
    # Static Chebyshev filter
    # -------------------------------------------------------------------------
    cheb_static_filter_1 = ChebyshevQuantumGraphFilter(
        coefficients=[0.0, 1.0, 0.25],
        rescale_laplacian=True,
    )
    cheb_static_filter_2 = ChebyshevQuantumGraphFilter(
        coefficients=[0.0, 1.0, 0.25],
        rescale_laplacian=True,
    )

    cheb_static_model = build_two_layer_quantum_model(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        first_filter=cheb_static_filter_1,
        second_filter=cheb_static_filter_2,
        hidden_activation=relu,
        output_activation=identity,
        random_state=random_state,
        W1=W1,
        W2=W2,
    )

    H1_cheb_static = cheb_static_model.layer1.forward(graph_input)
    H2_cheb_static = cheb_static_model.layer2.forward(graph_input, features=H1_cheb_static)
    model_outputs["Chebyshev static"] = (H1_cheb_static, H2_cheb_static)

    print_model_summary(
        "Static Chebyshev spectral filter",
        H1_cheb_static,
        H2_cheb_static,
        H1_classical,
        H2_classical,
    )

    # -------------------------------------------------------------------------
    # Quantum-controlled Chebyshev filter
    # -------------------------------------------------------------------------
    cheb_provider = QiskitVariationalCoefficientProvider(
        num_qubits=3,
        num_layers=2,
        input_scale=0.5,
        output_scale=0.5,
        output_bias=np.array([0.0, 1.0, 0.0]),
    )

    print_generated_coefficients(
        title="Quantum-generated Chebyshev coefficients on input features",
        provider=cheb_provider,
        graph_input=graph_input,
        features=np.asarray(graph_input.node_signals, dtype=float),
        num_coefficients=3,
    )

    cheb_quantum_filter_1 = ChebyshevQuantumGraphFilter(
        coefficients=[0.0, 1.0, 0.0],
        rescale_laplacian=True,
        coefficient_provider=cheb_provider,
    )
    cheb_quantum_filter_2 = ChebyshevQuantumGraphFilter(
        coefficients=[0.0, 1.0, 0.0],
        rescale_laplacian=True,
        coefficient_provider=cheb_provider,
    )

    cheb_quantum_model = build_two_layer_quantum_model(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        first_filter=cheb_quantum_filter_1,
        second_filter=cheb_quantum_filter_2,
        hidden_activation=relu,
        output_activation=identity,
        random_state=random_state,
        W1=W1,
        W2=W2,
    )

    H1_cheb_quantum = cheb_quantum_model.layer1.forward(graph_input)

    print_generated_coefficients(
        title="Quantum-generated Chebyshev coefficients after layer 1",
        provider=cheb_provider,
        graph_input=graph_input,
        features=H1_cheb_quantum,
        num_coefficients=3,
    )

    H2_cheb_quantum = cheb_quantum_model.layer2.forward(
        graph_input,
        features=H1_cheb_quantum,
    )
    model_outputs["Chebyshev quantum"] = (H1_cheb_quantum, H2_cheb_quantum)

    print_model_summary(
        "Quantum-controlled Chebyshev spectral filter",
        H1_cheb_quantum,
        H2_cheb_quantum,
        H1_classical,
        H2_classical,
    )

    # -------------------------------------------------------------------------
    # Exponential quantum filter
    # -------------------------------------------------------------------------
    exp_filter_1 = ExponentialQuantumGraphFilter(alpha=0.5, rescale_laplacian=True)
    exp_filter_2 = ExponentialQuantumGraphFilter(alpha=0.5, rescale_laplacian=True)

    exp_model = build_two_layer_quantum_model(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        first_filter=exp_filter_1,
        second_filter=exp_filter_2,
        hidden_activation=relu,
        output_activation=identity,
        random_state=random_state,
        W1=W1,
        W2=W2,
    )

    H1_exp = exp_model.layer1.forward(graph_input)
    H2_exp = exp_model.layer2.forward(graph_input, features=H1_exp)
    model_outputs["Exponential"] = (H1_exp, H2_exp)

    print_model_summary(
        "Exponential quantum filter",
        H1_exp,
        H2_exp,
        H1_classical,
        H2_classical,
    )

    # -------------------------------------------------------------------------
    # Static polynomial block-encoding-inspired filter
    # -------------------------------------------------------------------------
    block_static_filter_1 = PolynomialBlockEncodingQuantumGraphFilter(
        coefficients=[0.0, 1.0, 0.25],
        rescale_laplacian=True,
        coefficient_provider=None,
        operator_scale=None,
        renormalize_by_success_probability=False,
    )
    block_static_filter_2 = PolynomialBlockEncodingQuantumGraphFilter(
        coefficients=[0.0, 1.0, 0.25],
        rescale_laplacian=True,
        coefficient_provider=None,
        operator_scale=None,
        renormalize_by_success_probability=False,
    )

    block_static_model = build_two_layer_quantum_model(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        first_filter=block_static_filter_1,
        second_filter=block_static_filter_2,
        hidden_activation=relu,
        output_activation=identity,
        random_state=random_state,
        W1=W1,
        W2=W2,
    )

    H1_block_static = block_static_model.layer1.forward(graph_input)
    H2_block_static = block_static_model.layer2.forward(
        graph_input,
        features=H1_block_static,
    )
    model_outputs["Block polynomial static"] = (H1_block_static, H2_block_static)

    print_model_summary(
        "Static polynomial block-encoding-inspired filter",
        H1_block_static,
        H2_block_static,
        H1_classical,
        H2_classical,
    )

    # -------------------------------------------------------------------------
    # Quantum-controlled polynomial block-encoding-inspired filter
    # -------------------------------------------------------------------------
    block_provider = QiskitVariationalCoefficientProvider(
        num_qubits=3,
        num_layers=2,
        input_scale=0.5,
        output_scale=0.5,
        output_bias=np.array([0.0, 1.0, 0.0]),
    )

    print_generated_coefficients(
        title="Quantum-generated block-polynomial coefficients on input features",
        provider=block_provider,
        graph_input=graph_input,
        features=np.asarray(graph_input.node_signals, dtype=float),
        num_coefficients=3,
    )

    block_quantum_filter_1 = PolynomialBlockEncodingQuantumGraphFilter(
        coefficients=[0.0, 1.0, 0.0],
        rescale_laplacian=True,
        coefficient_provider=block_provider,
        operator_scale=None,
        renormalize_by_success_probability=False,
    )
    block_quantum_filter_2 = PolynomialBlockEncodingQuantumGraphFilter(
        coefficients=[0.0, 1.0, 0.0],
        rescale_laplacian=True,
        coefficient_provider=block_provider,
        operator_scale=None,
        renormalize_by_success_probability=False,
    )

    block_quantum_model = build_two_layer_quantum_model(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        first_filter=block_quantum_filter_1,
        second_filter=block_quantum_filter_2,
        hidden_activation=relu,
        output_activation=identity,
        random_state=random_state,
        W1=W1,
        W2=W2,
    )

    H1_block_quantum = block_quantum_model.layer1.forward(graph_input)

    print_generated_coefficients(
        title="Quantum-generated block-polynomial coefficients after layer 1",
        provider=block_provider,
        graph_input=graph_input,
        features=H1_block_quantum,
        num_coefficients=3,
    )

    H2_block_quantum = block_quantum_model.layer2.forward(
        graph_input,
        features=H1_block_quantum,
    )
    model_outputs["Block polynomial quantum"] = (H1_block_quantum, H2_block_quantum)

    print_model_summary(
        "Quantum-controlled polynomial block-encoding-inspired filter",
        H1_block_quantum,
        H2_block_quantum,
        H1_classical,
        H2_classical,
    )

    # -------------------------------------------------------------------------
    # Mimic / approximation quantum filter
    # -------------------------------------------------------------------------
    mimic_filter_1 = MimicQuantumGraphFilter(
        evolution_times=[0.10, 0.25, 0.50],
        mixture_coefficients=[1.0, -0.35, 0.10],
        residual_coefficient=0.0,
        rescale_laplacian=True,
        take_real_part=True,
        subtract_identity_from_evolution=True,
    )
    mimic_filter_2 = MimicQuantumGraphFilter(
        evolution_times=[0.10, 0.25, 0.50],
        mixture_coefficients=[1.0, -0.35, 0.10],
        residual_coefficient=0.0,
        rescale_laplacian=True,
        take_real_part=True,
        subtract_identity_from_evolution=True,
    )

    mimic_model = build_two_layer_quantum_model(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        first_filter=mimic_filter_1,
        second_filter=mimic_filter_2,
        hidden_activation=relu,
        output_activation=identity,
        random_state=random_state,
        W1=W1,
        W2=W2,
    )

    H1_mimic = mimic_model.layer1.forward(graph_input)
    H2_mimic = mimic_model.layer2.forward(graph_input, features=H1_mimic)
    model_outputs["Mimic quantum"] = (H1_mimic, H2_mimic)

    print_model_summary(
        "Mimic / approximation quantum filter",
        H1_mimic,
        H2_mimic,
        H1_classical,
        H2_classical,
    )

    # -------------------------------------------------------------------------
    # Cross-summary
    # -------------------------------------------------------------------------
    print_cross_summary(
        baseline_name="Classical",
        baseline_h1=H1_classical,
        baseline_h2=H2_classical,
        model_outputs=model_outputs,
    )

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    hidden_dict = {name: outputs[0] for name, outputs in model_outputs.items()}
    plot_all_hidden_comparisons(
        G=G,
        graph_input=graph_input,
        model_hidden_dict=hidden_dict,
        baseline_key="Classical",
    )


if __name__ == "__main__":
    main()
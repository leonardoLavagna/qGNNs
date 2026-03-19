import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from utils.graphs import *
from quantum_gnns.qgtheta import *



# ---------------------------------------------------------------------
# Classical reference implementation
# ---------------------------------------------------------------------
class ClassicalGraphLayer:
    """Simple classical Kipf-Welling-style graph layer.

    Applies:
        H_out = activation(A_hat @ H_in @ W + b)

    Notes:
        - Bias is optional.
        - This is used only as a reference baseline for comparison.
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
        out = a_hat @ features @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return self.activation(out)


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


# ---------------------------------------------------------------------
# Small helper utilities
# ---------------------------------------------------------------------
def print_model_comparison(
    name: str,
    h1_model: np.ndarray,
    h2_model: np.ndarray,
    h1_classical: np.ndarray,
    h2_classical: np.ndarray,
) -> None:
    """Print numerical comparisons against the classical baseline."""
    print("\n" + "=" * 80)
    print(f"{name}")
    print("=" * 80)
    print("Hidden representation H1:\n", h1_model)
    print("H1 shape:", h1_model.shape)
    print("Final representation H2:\n", h2_model)
    print("H2 shape:", h2_model.shape)

    diff_h1 = h1_model - h1_classical
    diff_h2 = h2_model - h2_classical

    print("\nDifference H1 (model - classical):\n", diff_h1)
    print("Difference H2 (model - classical):\n", diff_h2)
    print("||H1_model - H1_classical||_F =", np.linalg.norm(diff_h1))
    print("||H2_model - H2_classical||_F =", np.linalg.norm(diff_h2))


def plot_hidden_state_comparison(
    G: nx.Graph,
    graph_input,
    h1_classical: np.ndarray,
    h1_model: np.ndarray,
    title_prefix: str,
) -> None:
    """Plot classical vs model hidden node channels and their difference."""
    pos = nx.spring_layout(G, seed=42)

    edge_values = []
    for u, v in G.edges():
        key = (u, v) if (u, v) in graph_input.edge_signals else (v, u)
        edge_signal = graph_input.edge_signals[key]
        if isinstance(edge_signal, (list, tuple, np.ndarray)):
            edge_values.append(float(edge_signal[0]))
        else:
            edge_values.append(float(edge_signal))

    num_hidden_channels = h1_classical.shape[1]

    all_hidden_vals = np.concatenate([h1_classical.reshape(-1), h1_model.reshape(-1)])
    vmin_hidden, vmax_hidden = all_hidden_vals.min(), all_hidden_vals.max()

    all_diff_vals = (h1_model - h1_classical).reshape(-1)
    dmax_global = np.max(np.abs(all_diff_vals))
    if dmax_global < 1e-12:
        dmax_global = 1.0

    fig, axes = plt.subplots(3, num_hidden_channels, figsize=(5 * num_hidden_channels, 12))

    if num_hidden_channels == 1:
        axes = np.array(axes).reshape(3, 1)

    for j in range(num_hidden_channels):
        hidden_classical = h1_classical[:, j]
        hidden_model = h1_model[:, j]
        diff = hidden_model - hidden_classical

        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color=hidden_classical,
            edge_color=edge_values,
            cmap="viridis",
            edge_cmap=plt.cm.plasma,
            vmin=vmin_hidden,
            vmax=vmax_hidden,
            node_size=700,
            ax=axes[0, j],
        )
        axes[0, j].set_title(f"Classical ch {j}")

        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color=hidden_model,
            edge_color=edge_values,
            cmap="viridis",
            edge_cmap=plt.cm.plasma,
            vmin=vmin_hidden,
            vmax=vmax_hidden,
            node_size=700,
            ax=axes[1, j],
        )
        axes[1, j].set_title(f"{title_prefix} ch {j}")

        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color=diff,
            edge_color=edge_values,
            cmap="coolwarm",
            edge_cmap=plt.cm.plasma,
            vmin=-dmax_global,
            vmax=dmax_global,
            node_size=700,
            ax=axes[2, j],
        )
        axes[2, j].set_title(f"Difference ch {j}")

    fig.suptitle(f"Hidden-state comparison: Classical vs {title_prefix}", fontsize=14)
    plt.tight_layout()
    plt.show()


def build_demo_graph_input():
    """Create a small demo graph and associated signals."""
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


def main() -> None:
    # -----------------------------------------------------------------
    # Build graph input
    # -----------------------------------------------------------------
    G, graph_input = build_demo_graph_input()

    print("=" * 80)
    print("Graph input")
    print("=" * 80)
    print("Node order:", graph_input.node_order)
    print("Node signals:\n", graph_input.node_signals)
    print("Kipf-Welling adjacency A_hat:\n", graph_input.kipf_welling_adjacency())

    # -----------------------------------------------------------------
    # Shared architecture settings
    # -----------------------------------------------------------------
    in_features = 2
    hidden_features = 3
    out_features = 1
    random_state = 0

    # -----------------------------------------------------------------
    # Classical baseline
    # -----------------------------------------------------------------
    classical_model = ClassicalTwoLayerGraphNetwork(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        hidden_activation=relu,
        output_activation=identity,
        random_state=random_state,
    )

    # Save weights for fair comparison
    W1 = classical_model.layer1.weight.copy()
    W2 = classical_model.layer2.weight.copy()

    H1_classical, H2_classical = classical_model.forward(graph_input)

    print("\n" + "=" * 80)
    print("Classical baseline")
    print("=" * 80)
    print("Classical hidden representation H1:\n", H1_classical)
    print("H1_classical shape:", H1_classical.shape)
    print("Classical final representation H2:\n", H2_classical)
    print("H2_classical shape:", H2_classical.shape)

    # -----------------------------------------------------------------
    # Option 1: First-order spectral g_theta
    # g_theta(X) = theta_0 X + theta_1 A_hat X
    # Use theta_0=0, theta_1=1 to match classical propagation closely.
    # -----------------------------------------------------------------
    first_filter_1 = FirstOrderQuantumGraphFilter(theta_0=0.0, theta_1=1.0)
    first_filter_2 = FirstOrderQuantumGraphFilter(theta_0=0.0, theta_1=1.0)

    first_order_model = TwoLayerQuantumGraphNetwork(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        first_filter=first_filter_1,
        second_filter=first_filter_2,
        hidden_activation=relu,
        output_activation=identity,
        random_state=random_state,
    )

    first_order_model.layer1.weight = W1.copy()
    first_order_model.layer2.weight = W2.copy()
    first_order_model.layer1.bias = None
    first_order_model.layer2.bias = None

    H1_first_order = first_order_model.layer1.forward(graph_input)
    H2_first_order = first_order_model.layer2.forward(graph_input, features=H1_first_order)

    print_model_comparison(
        name="First-order spectral g_theta model",
        h1_model=H1_first_order,
        h2_model=H2_first_order,
        h1_classical=H1_classical,
        h2_classical=H2_classical,
    )

    # -----------------------------------------------------------------
    # Option 2: Chebyshev spectral g_theta
    # Try degree-2 filter as an example.
    # -----------------------------------------------------------------
    cheb_filter_1 = ChebyshevQuantumGraphFilter(
        coefficients=[0.0, 1.0, 0.25],
        rescale_laplacian=True,
    )
    cheb_filter_2 = ChebyshevQuantumGraphFilter(
        coefficients=[0.0, 1.0, 0.25],
        rescale_laplacian=True,
    )

    cheb_model = TwoLayerQuantumGraphNetwork(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        first_filter=cheb_filter_1,
        second_filter=cheb_filter_2,
        hidden_activation=relu,
        output_activation=identity,
        random_state=random_state,
    )

    cheb_model.layer1.weight = W1.copy()
    cheb_model.layer2.weight = W2.copy()
    cheb_model.layer1.bias = None
    cheb_model.layer2.bias = None

    H1_cheb = cheb_model.layer1.forward(graph_input)
    H2_cheb = cheb_model.layer2.forward(graph_input, features=H1_cheb)

    print_model_comparison(
        name="Chebyshev spectral g_theta model",
        h1_model=H1_cheb,
        h2_model=H2_cheb,
        h1_classical=H1_classical,
        h2_classical=H2_classical,
    )

    # -----------------------------------------------------------------
    # Optional: original exponential Laplacian-evolution model
    # Included only for comparison with the old implementation.
    # -----------------------------------------------------------------
    exp_filter_1 = ExponentialQuantumGraphFilter(
        alpha=0.5,
        rescale_laplacian=True,
    )
    exp_filter_2 = ExponentialQuantumGraphFilter(
        alpha=0.5,
        rescale_laplacian=True,
    )

    exp_model = TwoLayerQuantumGraphNetwork(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        first_filter=exp_filter_1,
        second_filter=exp_filter_2,
        hidden_activation=relu,
        output_activation=identity,
        random_state=random_state,
    )

    exp_model.layer1.weight = W1.copy()
    exp_model.layer2.weight = W2.copy()
    exp_model.layer1.bias = None
    exp_model.layer2.bias = None

    H1_exp = exp_model.layer1.forward(graph_input)
    H2_exp = exp_model.layer2.forward(graph_input, features=H1_exp)

    print_model_comparison(
        name="Exponential Laplacian-evolution model",
        h1_model=H1_exp,
        h2_model=H2_exp,
        h1_classical=H1_classical,
        h2_classical=H2_classical,
    )

    # -----------------------------------------------------------------
    # Cross-model summary
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Cross-model summary")
    print("=" * 80)
    print("||H1_first_order - H1_classical||_F =", np.linalg.norm(H1_first_order - H1_classical))
    print("||H2_first_order - H2_classical||_F =", np.linalg.norm(H2_first_order - H2_classical))
    print("||H1_cheb - H1_classical||_F =", np.linalg.norm(H1_cheb - H1_classical))
    print("||H2_cheb - H2_classical||_F =", np.linalg.norm(H2_cheb - H2_classical))
    print("||H1_exp - H1_classical||_F =", np.linalg.norm(H1_exp - H1_classical))
    print("||H2_exp - H2_classical||_F =", np.linalg.norm(H2_exp - H2_classical))
    print("||H1_cheb - H1_first_order||_F =", np.linalg.norm(H1_cheb - H1_first_order))
    print("||H2_cheb - H2_first_order||_F =", np.linalg.norm(H2_cheb - H2_first_order))

    # -----------------------------------------------------------------
    # Visualization
    # -----------------------------------------------------------------
    plot_hidden_state_comparison(
        G=G,
        graph_input=graph_input,
        h1_classical=H1_classical,
        h1_model=H1_first_order,
        title_prefix="First-order",
    )

    plot_hidden_state_comparison(
        G=G,
        graph_input=graph_input,
        h1_classical=H1_classical,
        h1_model=H1_cheb,
        title_prefix="Chebyshev",
    )

    plot_hidden_state_comparison(
        G=G,
        graph_input=graph_input,
        h1_classical=H1_classical,
        h1_model=H1_exp,
        title_prefix="Exponential",
    )


if __name__ == "__main__":
    main()
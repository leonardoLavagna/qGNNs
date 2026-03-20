# -----------------------------------------------------------------------------
# demo_grid_hybrid_gnn.py
#
# End-to-end demo for a hybrid graph neural network workflow on a small
# power-grid task using:
# - pandapower to generate supervised labels from power-flow simulations,
# - classical_gnns.shallow_kipf_welling_gnn as a classical benchmark,
# - quantum_gnns.qgtheta as the hybrid / quantum-inspired alternatives.
#
# Task:
#   Node-level regression of bus voltage magnitudes (vm_pu) from node features
#   representing power injections at each bus.
#
# Training:
#   - fixed graph topology
#   - multiple operating scenarios
#   - train on a scenario set
#   - evaluate on held-out scenarios
#
# Notes:
#   - this is a compact research demo, not an optimized production trainer
#   - optimization uses scipy.optimize.minimize on flattened parameters
#   - static filters train linear layer weights/biases
#   - quantum-controlled filters also train provider circuit parameters
#   - all figures, tables, and key values are automatically saved
#
# Expected project structure:
# - utils/graphs.py
# - classical_gnns/shallow_kipf_welling_gnn.py
# - quantum_gnns/qgtheta.py
#
# Optional extra dependency:
#   pip install pandapower pandas scipy networkx matplotlib
#
# © Leonardo Lavagna 2026
# -----------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
from pathlib import Path
from datetime import datetime
import copy
import json
import warnings

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import pandapower as pp

from utils.graphs import build_graph_input
from classical_gnns.shallow_kipf_welling_gnn import (
    KipfWellingLayer,
    identity,
    relu,
)
from quantum_gnns.qgtheta import (
    QiskitVariationalCoefficientProvider,
    ExponentialQuantumGraphFilter,
    FirstOrderQuantumGraphFilter,
    ChebyshevQuantumGraphFilter,
    PolynomialBlockEncodingQuantumGraphFilter,
    MimicQuantumGraphFilter,
    TwoLayerQuantumGraphNetwork,
)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def print_block(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def log_block(log_lines: list[str], title: str) -> None:
    log_lines.append("\n" + "=" * 100)
    log_lines.append(title)
    log_lines.append("=" * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def to_serializable(obj):
    """
    Convert numpy-heavy objects to JSON-serializable Python objects.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj


def create_output_dir(base_dir: str = "demo_outputs") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"demo_grid_hybrid_gnn_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    (output_dir / "arrays").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    return output_dir


def save_json(data, filepath: Path) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(to_serializable(data), f, indent=2)


def save_numpy_array(array: np.ndarray, filepath: Path) -> None:
    np.save(filepath, np.asarray(array))


def save_dataframe(df: pd.DataFrame, filepath: Path) -> None:
    df.to_csv(filepath, index=False)


def save_text(text: str, filepath: Path) -> None:
    filepath.write_text(text, encoding="utf-8")


def save_current_figure(filepath: Path) -> None:
    plt.savefig(filepath, dpi=300, bbox_inches="tight")


# -----------------------------------------------------------------------------
# Power-grid dataset generation
# -----------------------------------------------------------------------------
def build_demo_pandapower_grid() -> pp.pandapowerNet:
    """
    Build a small radial 5-bus distribution-style network.
    """
    net = pp.create_empty_network(sn_mva=100.0)
    bus0 = pp.create_bus(net, vn_kv=20.0, name="Bus 0")
    bus1 = pp.create_bus(net, vn_kv=20.0, name="Bus 1")
    bus2 = pp.create_bus(net, vn_kv=20.0, name="Bus 2")
    bus3 = pp.create_bus(net, vn_kv=20.0, name="Bus 3")
    bus4 = pp.create_bus(net, vn_kv=20.0, name="Bus 4")
    pp.create_ext_grid(net, bus=bus0, vm_pu=1.00, name="Grid Connection")

    for bus in [bus1, bus2, bus3, bus4]:
        pp.create_load(net, bus=bus, p_mw=0.5, q_mvar=0.1, name=f"Load@{bus}")

    line_kwargs = dict(
        length_km=1.0,
        r_ohm_per_km=0.25,
        x_ohm_per_km=0.10,
        c_nf_per_km=10.0,
        max_i_ka=0.40,
    )

    pp.create_line_from_parameters(net, bus0, bus1, **line_kwargs, name="L01")
    pp.create_line_from_parameters(net, bus1, bus2, **line_kwargs, name="L12")
    pp.create_line_from_parameters(net, bus2, bus3, **line_kwargs, name="L23")
    pp.create_line_from_parameters(net, bus3, bus4, **line_kwargs, name="L34")

    return net


def build_graph_input_from_pandapower(
    net: pp.pandapowerNet,
    in_features: int,
) -> tuple[nx.Graph, object]:
    """
    Convert the static network topology into a GraphInput.

    Node signals are placeholders; scenario-specific features are passed later
    during forward calls.
    """
    G = nx.Graph()
    num_buses = len(net.bus)

    for bus_idx in range(num_buses):
        G.add_node(int(bus_idx))

    edge_signals: dict[tuple[int, int], list[float]] = {}
    for _, row in net.line.iterrows():
        u = int(row.from_bus)
        v = int(row.to_bus)
        G.add_edge(u, v)
        edge_signals[(u, v)] = [
            float(row.r_ohm_per_km),
            float(row.x_ohm_per_km),
            float(row.max_i_ka),
        ]

    dummy_node_signals = np.zeros((num_buses, in_features), dtype=float)
    graph_input = build_graph_input(
        G,
        node_signals=dummy_node_signals,
        edge_signals=edge_signals,
    )
    return G, graph_input


def sample_operating_scenario(
    net_template: pp.pandapowerNet,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create one random operating scenario, run a power flow, and return:

    X: node features of shape (N, 3)
       [p_mw, q_mvar, is_slack]
    Y: regression target of shape (N, 1)
       [vm_pu]
    """
    net = copy.deepcopy(net_template)

    for load_idx in net.load.index:
        p_mw = rng.uniform(0.20, 1.20)
        q_mvar = rng.uniform(0.05, 0.50)
        net.load.at[load_idx, "p_mw"] = float(p_mw)
        net.load.at[load_idx, "q_mvar"] = float(q_mvar)

    pp.runpp(net, algorithm="nr", init="flat")

    num_buses = len(net.bus)
    x = np.zeros((num_buses, 3), dtype=float)

    pq_by_bus = {int(bus): [0.0, 0.0] for bus in net.bus.index}
    for _, row in net.load.iterrows():
        bus = int(row.bus)
        pq_by_bus[bus][0] += float(row.p_mw)
        pq_by_bus[bus][1] += float(row.q_mvar)

    slack_buses = set(int(bus) for bus in net.ext_grid.bus.values)

    for bus in net.bus.index:
        bus = int(bus)
        p_mw, q_mvar = pq_by_bus[bus]
        is_slack = 1.0 if bus in slack_buses else 0.0
        x[bus, :] = [p_mw, q_mvar, is_slack]

    y = net.res_bus.vm_pu.to_numpy(dtype=float).reshape(-1, 1)
    return x, y


def generate_dataset(
    net_template: pp.pandapowerNet,
    num_samples: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset of operating scenarios.

    Returns
    -------
    X : (S, N, F)
    Y : (S, N, 1)
    """
    rng = np.random.default_rng(seed)
    xs = []
    ys = []
    for _ in range(num_samples):
        x, y = sample_operating_scenario(net_template, rng)
        xs.append(x)
        ys.append(y)
    return np.stack(xs, axis=0), np.stack(ys, axis=0)


# -----------------------------------------------------------------------------
# Classical two-layer model
# -----------------------------------------------------------------------------
@dataclass
class ClassicalTwoLayerModel:
    """
    Two-layer classical Kipf-Welling model with explicit NumPy parameters.
    """
    in_features: int
    hidden_features: int
    out_features: int

    def num_parameters(self) -> int:
        return (
            self.in_features * self.hidden_features
            + self.hidden_features
            + self.hidden_features * self.out_features
            + self.out_features
        )

    def unpack(
        self,
        params: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        params = np.asarray(params, dtype=float)
        idx = 0

        w1_size = self.in_features * self.hidden_features
        b1_size = self.hidden_features
        w2_size = self.hidden_features * self.out_features
        b2_size = self.out_features

        w1 = params[idx: idx + w1_size].reshape(self.in_features, self.hidden_features)
        idx += w1_size

        b1 = params[idx: idx + b1_size]
        idx += b1_size

        w2 = params[idx: idx + w2_size].reshape(self.hidden_features, self.out_features)
        idx += w2_size

        b2 = params[idx: idx + b2_size]
        return w1, b1, w2, b2

    def forward(
        self,
        graph_input,
        features: np.ndarray,
        params: np.ndarray,
    ) -> np.ndarray:
        w1, b1, w2, b2 = self.unpack(params)

        layer1 = KipfWellingLayer(weight=w1, bias=b1, activation=relu)
        layer2 = KipfWellingLayer(weight=w2, bias=b2, activation=identity)

        h1 = layer1.forward(graph_input=graph_input, features=features)
        h2 = layer2.forward(graph_input=graph_input, features=h1)
        return h2


# -----------------------------------------------------------------------------
# Quantum / hybrid model wrappers
# -----------------------------------------------------------------------------
@dataclass
class QuantumModelSpec:
    """
    Wrapper around TwoLayerQuantumGraphNetwork with flattened parameters.
    """
    name: str
    in_features: int
    hidden_features: int
    out_features: int
    filter_builder: Callable[[Optional[np.ndarray]], tuple[object, object]]
    provider_num_parameters: int = 0

    def num_parameters(self) -> int:
        layer_params = (
            self.in_features * self.hidden_features
            + self.hidden_features
            + self.hidden_features * self.out_features
            + self.out_features
        )
        return layer_params + self.provider_num_parameters

    def unpack(self, params: np.ndarray):
        params = np.asarray(params, dtype=float)
        idx = 0

        w1_size = self.in_features * self.hidden_features
        b1_size = self.hidden_features
        w2_size = self.hidden_features * self.out_features
        b2_size = self.out_features

        w1 = params[idx: idx + w1_size].reshape(self.in_features, self.hidden_features)
        idx += w1_size

        b1 = params[idx: idx + b1_size]
        idx += b1_size

        w2 = params[idx: idx + w2_size].reshape(self.hidden_features, self.out_features)
        idx += w2_size

        b2 = params[idx: idx + b2_size]
        idx += b2_size

        provider_params = None
        if self.provider_num_parameters > 0:
            provider_params = params[idx: idx + self.provider_num_parameters]
            idx += self.provider_num_parameters

        return w1, b1, w2, b2, provider_params

    def forward(
        self,
        graph_input,
        features: np.ndarray,
        params: np.ndarray,
    ) -> np.ndarray:
        w1, b1, w2, b2, provider_params = self.unpack(params)
        first_filter, second_filter = self.filter_builder(provider_params)

        model = TwoLayerQuantumGraphNetwork(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            first_filter=first_filter,
            second_filter=second_filter,
            hidden_activation=relu,
            output_activation=identity,
            random_state=0,
        )

        model.layer1.weight = w1
        model.layer1.bias = b1
        model.layer2.weight = w2
        model.layer2.bias = b2

        return model.forward(graph_input=graph_input, features=features)


# -----------------------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------------------
def build_initial_parameter_vector(
    num_parameters: int,
    seed: int = 0,
    scale: float = 0.15,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=scale, size=num_parameters)


def dataset_loss(
    params: np.ndarray,
    model_forward: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x_data: np.ndarray,
    y_data: np.ndarray,
    l2_weight: float = 1e-4,
) -> float:
    total = 0.0
    num_samples = x_data.shape[0]
    for x, y in zip(x_data, y_data):
        y_pred = model_forward(x, params)
        total += np.mean((y_pred - y) ** 2)
    total /= max(num_samples, 1)
    total += l2_weight * float(np.mean(params ** 2))
    return float(total)


def predict_dataset(
    model_forward: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x_data: np.ndarray,
    params: np.ndarray,
) -> np.ndarray:
    preds = [model_forward(x, params) for x in x_data]
    return np.stack(preds, axis=0)


def train_model(
    name: str,
    model_forward: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x_train: np.ndarray,
    y_train: np.ndarray,
    init_params: np.ndarray,
    maxiter: int = 120,
) -> np.ndarray:
    print_block(f"Training {name}")

    def objective(p: np.ndarray) -> float:
        return dataset_loss(
            params=p,
            model_forward=model_forward,
            x_data=x_train,
            y_data=y_train,
            l2_weight=1e-4,
        )

    result = minimize(
        objective,
        x0=init_params,
        method="L-BFGS-B",
        options={"maxiter": maxiter},
    )

    print(f"{name} final training objective: {result.fun:.8f}")
    return np.asarray(result.x, dtype=float)


# -----------------------------------------------------------------------------
# Message-passing comparison helpers
# -----------------------------------------------------------------------------
def get_classical_message_passing(graph_input, features: np.ndarray) -> np.ndarray:
    """
    Classical Kipf-Welling message passing before the learned linear map:
        A_hat @ X
    """
    a_hat = np.asarray(graph_input.kipf_welling_adjacency(), dtype=float)
    return a_hat @ np.asarray(features, dtype=float)


def get_filter_message_passing(
    graph_input,
    features: np.ndarray,
    graph_filter,
) -> np.ndarray:
    """
    Apply a qgtheta graph filter directly to features, i.e. compare the
    propagation/message-passing stage before the learned linear map.
    """
    return graph_filter.forward(graph_input, np.asarray(features, dtype=float))


def build_edge_values_for_plot(graph_input, G) -> list[float]:
    """
    Extract one scalar per edge for coloring.
    """
    edge_values = []
    for u, v in G.edges():
        key = (u, v) if (u, v) in graph_input.edge_signals else (v, u)
        edge_signal = graph_input.edge_signals[key]
        if isinstance(edge_signal, (list, tuple, np.ndarray)):
            edge_values.append(float(edge_signal[0]))
        else:
            edge_values.append(float(edge_signal))
    return edge_values


def plot_message_passing_line_comparison(
    graph_input,
    message_dict: dict[str, np.ndarray],
    feature_index: int = 0,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Plot node-wise propagated values for one selected feature channel.
    """
    plt.figure(figsize=(11, 5))
    x_axis = np.arange(graph_input.num_nodes)

    for name, values in message_dict.items():
        plt.plot(
            x_axis,
            values[:, feature_index],
            marker="o",
            label=name,
        )

    plt.xticks(x_axis, [f"Bus {i}" for i in x_axis])
    plt.xlabel("Node")
    plt.ylabel(f"Propagated feature channel {feature_index}")
    plt.title("Message-passing comparison across models")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_current_figure(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_message_passing_graph_comparison(
    G,
    graph_input,
    message_dict: dict[str, np.ndarray],
    feature_index: int = 0,
    layout_seed: int = 42,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Plot graph-colored node values for one propagated feature channel.
    """
    pos = nx.spring_layout(G, seed=layout_seed)
    edge_values = build_edge_values_for_plot(graph_input, G)

    names = list(message_dict.keys())
    arrays = [message_dict[name][:, feature_index] for name in names]
    all_vals = np.concatenate(arrays)
    vmin = float(np.min(all_vals))
    vmax = float(np.max(all_vals))

    ncols = min(3, len(names))
    nrows = int(np.ceil(len(names) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    for ax_idx, name in enumerate(names):
        row = ax_idx // ncols
        col = ax_idx % ncols
        ax = axes[row, col]

        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color=message_dict[name][:, feature_index],
            edge_color=edge_values,
            cmap="viridis",
            edge_cmap=plt.cm.plasma,
            vmin=vmin,
            vmax=vmax,
            node_size=800,
            ax=ax,
        )
        ax.set_title(f"{name}\nfeature {feature_index}")

    for ax_idx in range(len(names), nrows * ncols):
        row = ax_idx // ncols
        col = ax_idx % ncols
        axes[row, col].axis("off")

    fig.suptitle("Graph message-passing comparison", fontsize=15)
    plt.tight_layout()

    if save_path is not None:
        save_current_figure(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_message_passing_difference_to_classical(
    G,
    graph_input,
    classical_message: np.ndarray,
    message_dict: dict[str, np.ndarray],
    feature_index: int = 0,
    layout_seed: int = 42,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Plot graph-colored differences relative to classical message passing.
    """
    pos = nx.spring_layout(G, seed=layout_seed)
    edge_values = build_edge_values_for_plot(graph_input, G)

    diff_names = [name for name in message_dict.keys() if name != "Classical A_hat @ X"]
    diffs = [
        message_dict[name][:, feature_index] - classical_message[:, feature_index]
        for name in diff_names
    ]

    if len(diffs) == 0:
        return

    all_diff_vals = np.concatenate(diffs)
    dmax = float(np.max(np.abs(all_diff_vals)))
    if dmax < 1e-12:
        dmax = 1.0

    ncols = min(3, len(diff_names))
    nrows = int(np.ceil(len(diff_names) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    for ax_idx, name in enumerate(diff_names):
        row = ax_idx // ncols
        col = ax_idx % ncols
        ax = axes[row, col]

        diff_vals = message_dict[name][:, feature_index] - classical_message[:, feature_index]

        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color=diff_vals,
            edge_color=edge_values,
            cmap="coolwarm",
            edge_cmap=plt.cm.plasma,
            vmin=-dmax,
            vmax=dmax,
            node_size=800,
            ax=ax,
        )
        ax.set_title(f"{name} - Classical\nfeature {feature_index}")

    for ax_idx in range(len(diff_names), nrows * ncols):
        row = ax_idx // ncols
        col = ax_idx % ncols
        axes[row, col].axis("off")

    fig.suptitle("Difference from classical message passing", fontsize=15)
    plt.tight_layout()

    if save_path is not None:
        save_current_figure(save_path)
    if show:
        plt.show()
    else:
        plt.close()


# -----------------------------------------------------------------------------
# Model factory
# -----------------------------------------------------------------------------
def make_model_specs(
    in_features: int,
    hidden_features: int,
    out_features: int,
) -> dict[str, object]:
    """
    Create all model specs to compare.
    """

    def first_static_builder(_provider_params):
        return (
            FirstOrderQuantumGraphFilter(theta_0=0.0, theta_1=1.0),
            FirstOrderQuantumGraphFilter(theta_0=0.0, theta_1=1.0),
        )

    first_provider_template = QiskitVariationalCoefficientProvider(
        num_qubits=2,
        num_layers=2,
        input_scale=0.5,
        output_scale=0.30,
        output_bias=np.array([0.0, 1.0]),
    )

    def first_quantum_builder(provider_params):
        p1 = QiskitVariationalCoefficientProvider(
            num_qubits=2,
            num_layers=2,
            input_scale=0.5,
            output_scale=0.30,
            output_bias=np.array([0.0, 1.0]),
        )
        p2 = QiskitVariationalCoefficientProvider(
            num_qubits=2,
            num_layers=2,
            input_scale=0.5,
            output_scale=0.30,
            output_bias=np.array([0.0, 1.0]),
        )
        if provider_params is not None:
            p1.set_parameter_vector(provider_params)
            p2.set_parameter_vector(provider_params)
        return (
            FirstOrderQuantumGraphFilter(coefficient_provider=p1),
            FirstOrderQuantumGraphFilter(coefficient_provider=p2),
        )

    def cheb_static_builder(_provider_params):
        return (
            ChebyshevQuantumGraphFilter(coefficients=[0.0, 1.0, 0.25], rescale_laplacian=True),
            ChebyshevQuantumGraphFilter(coefficients=[0.0, 1.0, 0.25], rescale_laplacian=True),
        )

    cheb_provider_template = QiskitVariationalCoefficientProvider(
        num_qubits=3,
        num_layers=2,
        input_scale=0.5,
        output_scale=0.20,
        output_bias=np.array([0.0, 1.0, 0.0]),
    )

    def cheb_quantum_builder(provider_params):
        p1 = QiskitVariationalCoefficientProvider(
            num_qubits=3,
            num_layers=2,
            input_scale=0.5,
            output_scale=0.20,
            output_bias=np.array([0.0, 1.0, 0.0]),
        )
        p2 = QiskitVariationalCoefficientProvider(
            num_qubits=3,
            num_layers=2,
            input_scale=0.5,
            output_scale=0.20,
            output_bias=np.array([0.0, 1.0, 0.0]),
        )
        if provider_params is not None:
            p1.set_parameter_vector(provider_params)
            p2.set_parameter_vector(provider_params)
        return (
            ChebyshevQuantumGraphFilter(
                coefficients=[0.0, 1.0, 0.0],
                rescale_laplacian=True,
                coefficient_provider=p1,
            ),
            ChebyshevQuantumGraphFilter(
                coefficients=[0.0, 1.0, 0.0],
                rescale_laplacian=True,
                coefficient_provider=p2,
            ),
        )

    def exponential_builder(_provider_params):
        return (
            ExponentialQuantumGraphFilter(alpha=0.35, rescale_laplacian=True),
            ExponentialQuantumGraphFilter(alpha=0.35, rescale_laplacian=True),
        )

    def block_static_builder(_provider_params):
        return (
            PolynomialBlockEncodingQuantumGraphFilter(
                coefficients=[0.0, 1.0, 0.25],
                rescale_laplacian=True,
                operator_scale=None,
                renormalize_by_success_probability=False,
            ),
            PolynomialBlockEncodingQuantumGraphFilter(
                coefficients=[0.0, 1.0, 0.25],
                rescale_laplacian=True,
                operator_scale=None,
                renormalize_by_success_probability=False,
            ),
        )

    block_provider_template = QiskitVariationalCoefficientProvider(
        num_qubits=3,
        num_layers=2,
        input_scale=0.5,
        output_scale=0.20,
        output_bias=np.array([0.0, 1.0, 0.0]),
    )

    def block_quantum_builder(provider_params):
        p1 = QiskitVariationalCoefficientProvider(
            num_qubits=3,
            num_layers=2,
            input_scale=0.5,
            output_scale=0.20,
            output_bias=np.array([0.0, 1.0, 0.0]),
        )
        p2 = QiskitVariationalCoefficientProvider(
            num_qubits=3,
            num_layers=2,
            input_scale=0.5,
            output_scale=0.20,
            output_bias=np.array([0.0, 1.0, 0.0]),
        )
        if provider_params is not None:
            p1.set_parameter_vector(provider_params)
            p2.set_parameter_vector(provider_params)
        return (
            PolynomialBlockEncodingQuantumGraphFilter(
                coefficients=[0.0, 1.0, 0.0],
                rescale_laplacian=True,
                coefficient_provider=p1,
                operator_scale=None,
                renormalize_by_success_probability=False,
            ),
            PolynomialBlockEncodingQuantumGraphFilter(
                coefficients=[0.0, 1.0, 0.0],
                rescale_laplacian=True,
                coefficient_provider=p2,
                operator_scale=None,
                renormalize_by_success_probability=False,
            ),
        )

    def mimic_builder(_provider_params):
        return (
            MimicQuantumGraphFilter(
                evolution_times=[0.10, 0.25, 0.50],
                mixture_coefficients=[1.0, -0.35, 0.10],
                residual_coefficient=1.0,
                rescale_laplacian=True,
                take_real_part=True,
                subtract_identity_from_evolution=False,
            ),
            MimicQuantumGraphFilter(
                evolution_times=[0.10, 0.25, 0.50],
                mixture_coefficients=[1.0, -0.35, 0.10],
                residual_coefficient=1.0,
                rescale_laplacian=True,
                take_real_part=True,
                subtract_identity_from_evolution=False,
            ),
        )

    return {
        "Classical Kipf-Welling": ClassicalTwoLayerModel(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
        ),
        "First-order static": QuantumModelSpec(
            name="First-order static",
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            filter_builder=first_static_builder,
            provider_num_parameters=0,
        ),
        "First-order quantum": QuantumModelSpec(
            name="First-order quantum",
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            filter_builder=first_quantum_builder,
            provider_num_parameters=first_provider_template.num_parameters(),
        ),
        "Chebyshev static": QuantumModelSpec(
            name="Chebyshev static",
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            filter_builder=cheb_static_builder,
            provider_num_parameters=0,
        ),
        "Chebyshev quantum": QuantumModelSpec(
            name="Chebyshev quantum",
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            filter_builder=cheb_quantum_builder,
            provider_num_parameters=cheb_provider_template.num_parameters(),
        ),
        "Exponential": QuantumModelSpec(
            name="Exponential",
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            filter_builder=exponential_builder,
            provider_num_parameters=0,
        ),
        "Block polynomial static": QuantumModelSpec(
            name="Block polynomial static",
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            filter_builder=block_static_builder,
            provider_num_parameters=0,
        ),
        "Block polynomial quantum": QuantumModelSpec(
            name="Block polynomial quantum",
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            filter_builder=block_quantum_builder,
            provider_num_parameters=block_provider_template.num_parameters(),
        ),
        "Mimic quantum": QuantumModelSpec(
            name="Mimic quantum",
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            filter_builder=mimic_builder,
            provider_num_parameters=0,
        ),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    np.set_printoptions(precision=5, suppress=True)

    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module="scipy.optimize",
    )

    output_dir = create_output_dir()
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    arrays_dir = output_dir / "arrays"
    logs_dir = output_dir / "logs"

    log_lines: list[str] = []
    log_block(log_lines, "Output directory")
    log_lines.append(str(output_dir))
    print_block("Output directory")
    print(output_dir)

    # -------------------------------------------------------------------------
    # Problem setup
    # -------------------------------------------------------------------------
    in_features = 3
    hidden_features = 6
    out_features = 1

    net = build_demo_pandapower_grid()
    G, graph_input = build_graph_input_from_pandapower(net, in_features=in_features)

    print_block("Static graph information")
    print("Nodes:", list(G.nodes()))
    print("Edges:", list(G.edges()))
    print("A_hat:\n", graph_input.kipf_welling_adjacency())
    print("L_tilde:\n", graph_input.kipf_welling_laplacian())
    print("Rescaled L_tilde:\n", graph_input.rescaled_kipf_welling_laplacian())

    log_block(log_lines, "Static graph information")
    log_lines.append(f"Nodes: {list(G.nodes())}")
    log_lines.append(f"Edges: {list(G.edges())}")
    log_lines.append(f"A_hat:\n{graph_input.kipf_welling_adjacency()}")
    log_lines.append(f"L_tilde:\n{graph_input.kipf_welling_laplacian()}")
    log_lines.append(f"Rescaled L_tilde:\n{graph_input.rescaled_kipf_welling_laplacian()}")

    save_json(
        {
            "nodes": list(G.nodes()),
            "edges": list(G.edges()),
            "a_hat": np.asarray(graph_input.kipf_welling_adjacency(), dtype=float),
            "l_tilde": np.asarray(graph_input.kipf_welling_laplacian(), dtype=float),
            "rescaled_l_tilde": np.asarray(graph_input.rescaled_kipf_welling_laplacian(), dtype=float),
            "edge_signals": graph_input.edge_signals,
        },
        tables_dir / "static_graph_information.json",
    )

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    x_all, y_all = generate_dataset(net, num_samples=80, seed=123)

    num_train = 60
    x_train, y_train = x_all[:num_train], y_all[:num_train]
    x_test, y_test = x_all[num_train:], y_all[num_train:]

    print_block("Dataset")
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    log_block(log_lines, "Dataset")
    log_lines.append(f"x_train shape: {x_train.shape}")
    log_lines.append(f"y_train shape: {y_train.shape}")
    log_lines.append(f"x_test shape: {x_test.shape}")
    log_lines.append(f"y_test shape: {y_test.shape}")

    save_numpy_array(x_all, arrays_dir / "x_all.npy")
    save_numpy_array(y_all, arrays_dir / "y_all.npy")
    save_numpy_array(x_train, arrays_dir / "x_train.npy")
    save_numpy_array(y_train, arrays_dir / "y_train.npy")
    save_numpy_array(x_test, arrays_dir / "x_test.npy")
    save_numpy_array(y_test, arrays_dir / "y_test.npy")

    # -------------------------------------------------------------------------
    # Build models
    # -------------------------------------------------------------------------
    specs = make_model_specs(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
    )

    fitted_params: dict[str, np.ndarray] = {}
    results = []

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    for model_name, spec in specs.items():
        print_block(f"Model: {model_name}")
        log_block(log_lines, f"Model: {model_name}")

        if isinstance(spec, ClassicalTwoLayerModel):
            init_params = build_initial_parameter_vector(
                spec.num_parameters(),
                seed=0,
                scale=0.15,
            )

            forward = lambda features, params, spec=spec: spec.forward(
                graph_input=graph_input,
                features=features,
                params=params,
            )
            trained = train_model(
                name=model_name,
                model_forward=forward,
                x_train=x_train,
                y_train=y_train,
                init_params=init_params,
                maxiter=150,
            )
        else:
            init_params = build_initial_parameter_vector(
                spec.num_parameters(),
                seed=0,
                scale=0.12,
            )

            forward = lambda features, params, spec=spec: spec.forward(
                graph_input=graph_input,
                features=features,
                params=params,
            )
            trained = train_model(
                name=model_name,
                model_forward=forward,
                x_train=x_train,
                y_train=y_train,
                init_params=init_params,
                maxiter=120,
            )

        fitted_params[model_name] = trained
        save_numpy_array(trained, arrays_dir / f"trained_params_{model_name.replace(' ', '_').replace('-', '_')}.npy")

        y_pred_train = predict_dataset(forward, x_train, trained)
        y_pred_test = predict_dataset(forward, x_test, trained)

        save_numpy_array(
            y_pred_train,
            arrays_dir / f"y_pred_train_{model_name.replace(' ', '_').replace('-', '_')}.npy",
        )
        save_numpy_array(
            y_pred_test,
            arrays_dir / f"y_pred_test_{model_name.replace(' ', '_').replace('-', '_')}.npy",
        )

        train_rmse = rmse(y_train, y_pred_train)
        test_rmse = rmse(y_test, y_pred_test)
        test_mae = mae(y_test, y_pred_test)

        log_lines.append(f"{model_name} train_rmse: {train_rmse:.8f}")
        log_lines.append(f"{model_name} test_rmse: {test_rmse:.8f}")
        log_lines.append(f"{model_name} test_mae: {test_mae:.8f}")
        log_lines.append(f"{model_name} num_parameters: {trained.size}")

        results.append(
            {
                "model": model_name,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "test_mae": test_mae,
                "num_parameters": trained.size,
            }
        )

    # -------------------------------------------------------------------------
    # Results table
    # -------------------------------------------------------------------------
    results_df = pd.DataFrame(results).sort_values("test_rmse").reset_index(drop=True)

    print_block("Results")
    print(results_df)

    log_block(log_lines, "Results")
    log_lines.append(results_df.to_string(index=False))

    save_dataframe(results_df, tables_dir / "results_summary.csv")
    save_json(results_df.to_dict(orient="records"), tables_dir / "results_summary.json")

    # -------------------------------------------------------------------------
    # Inference on one held-out scenario
    # -------------------------------------------------------------------------
    scenario_index = 0
    x_infer = x_test[scenario_index]
    y_true = y_test[scenario_index]

    print_block("Inference on one held-out scenario")
    print("Input node features [p_mw, q_mvar, is_slack]:\n", x_infer)
    print("True bus voltages vm_pu:\n", y_true.reshape(-1))

    log_block(log_lines, "Inference on one held-out scenario")
    log_lines.append(f"Input node features [p_mw, q_mvar, is_slack]:\n{x_infer}")
    log_lines.append(f"True bus voltages vm_pu:\n{y_true.reshape(-1)}")

    save_numpy_array(x_infer, arrays_dir / "x_infer.npy")
    save_numpy_array(y_true, arrays_dir / "y_true.npy")

    inference_rows = []
    predictions_for_plot = {}

    for model_name, spec in specs.items():
        params = fitted_params[model_name]
        y_pred = spec.forward(graph_input=graph_input, features=x_infer, params=params)

        predictions_for_plot[model_name] = y_pred.reshape(-1)
        save_numpy_array(
            y_pred.reshape(-1),
            arrays_dir / f"y_pred_inference_{model_name.replace(' ', '_').replace('-', '_')}.npy",
        )

        inference_rows.append(
            {
                "model": model_name,
                "scenario_rmse": rmse(y_true, y_pred),
                "scenario_mae": mae(y_true, y_pred),
            }
        )

    infer_df = pd.DataFrame(inference_rows).sort_values("scenario_rmse").reset_index(drop=True)
    print(infer_df)

    log_block(log_lines, "Inference summary")
    log_lines.append(infer_df.to_string(index=False))

    save_dataframe(infer_df, tables_dir / "inference_summary.csv")
    save_json(infer_df.to_dict(orient="records"), tables_dir / "inference_summary.json")
    save_json(predictions_for_plot, tables_dir / "inference_predictions.json")

    # -------------------------------------------------------------------------
    # Simple plot for the best three models on one scenario
    # -------------------------------------------------------------------------
    top_models = infer_df["model"].tolist()[:3]

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(y_true)), y_true.reshape(-1), marker="o", label="True vm_pu")

    for model_name in top_models:
        plt.plot(
            range(len(y_true)),
            predictions_for_plot[model_name],
            marker="x",
            label=model_name,
        )

    plt.xticks(range(len(y_true)), [f"Bus {i}" for i in range(len(y_true))])
    plt.ylabel("Voltage magnitude (p.u.)")
    plt.title("Inference on one held-out operating scenario")
    plt.legend()
    plt.tight_layout()
    save_current_figure(figures_dir / "inference_top3_models.png")
    plt.show()

    # -------------------------------------------------------------------------
    # Message-passing comparison on one held-out scenario
    # -------------------------------------------------------------------------
    print_block("Message-passing comparison on one held-out scenario")
    log_block(log_lines, "Message-passing comparison on one held-out scenario")

    x_message = x_infer
    classical_message = get_classical_message_passing(graph_input, x_message)

    first_quantum_spec = specs["First-order quantum"]
    _, _, _, _, first_provider_params = first_quantum_spec.unpack(
        fitted_params["First-order quantum"]
    )
    first_quantum_filter_msg, _ = first_quantum_spec.filter_builder(first_provider_params)

    cheb_quantum_spec = specs["Chebyshev quantum"]
    _, _, _, _, cheb_provider_params = cheb_quantum_spec.unpack(
        fitted_params["Chebyshev quantum"]
    )
    cheb_quantum_filter_msg, _ = cheb_quantum_spec.filter_builder(cheb_provider_params)

    block_quantum_spec = specs["Block polynomial quantum"]
    _, _, _, _, block_provider_params = block_quantum_spec.unpack(
        fitted_params["Block polynomial quantum"]
    )
    block_quantum_filter_msg, _ = block_quantum_spec.filter_builder(block_provider_params)

    first_static_filter_msg, _ = specs["First-order static"].filter_builder(None)
    cheb_static_filter_msg, _ = specs["Chebyshev static"].filter_builder(None)
    exp_filter_msg, _ = specs["Exponential"].filter_builder(None)
    block_static_filter_msg, _ = specs["Block polynomial static"].filter_builder(None)
    mimic_filter_msg, _ = specs["Mimic quantum"].filter_builder(None)

    message_dict = {
        "Input X": np.asarray(x_message, dtype=float),
        "Classical A_hat @ X": classical_message,
        "First-order static": get_filter_message_passing(
            graph_input, x_message, first_static_filter_msg
        ),
        "First-order quantum": get_filter_message_passing(
            graph_input, x_message, first_quantum_filter_msg
        ),
        "Chebyshev static": get_filter_message_passing(
            graph_input, x_message, cheb_static_filter_msg
        ),
        "Chebyshev quantum": get_filter_message_passing(
            graph_input, x_message, cheb_quantum_filter_msg
        ),
        "Exponential": get_filter_message_passing(
            graph_input, x_message, exp_filter_msg
        ),
        "Block polynomial static": get_filter_message_passing(
            graph_input, x_message, block_static_filter_msg
        ),
        "Block polynomial quantum": get_filter_message_passing(
            graph_input, x_message, block_quantum_filter_msg
        ),
        "Mimic quantum": get_filter_message_passing(
            graph_input, x_message, mimic_filter_msg
        ),
    }

    for name, values in message_dict.items():
        print(f"{name} shape: {values.shape}")
        print(values)
        log_lines.append(f"{name} shape: {values.shape}")
        log_lines.append(str(values))
        save_numpy_array(
            values,
            arrays_dir / f"message_passing_{name.replace(' ', '_').replace('@', 'at').replace('-', '_')}.npy",
        )

    save_json(message_dict, tables_dir / "message_passing_values.json")

    # Feature 0 = p_mw
    plot_message_passing_line_comparison(
        graph_input=graph_input,
        message_dict=message_dict,
        feature_index=0,
        save_path=figures_dir / "message_passing_line_feature_0.png",
        show=True,
    )
    plot_message_passing_graph_comparison(
        G=G,
        graph_input=graph_input,
        message_dict=message_dict,
        feature_index=0,
        save_path=figures_dir / "message_passing_graph_feature_0.png",
        show=True,
    )
    plot_message_passing_difference_to_classical(
        G=G,
        graph_input=graph_input,
        classical_message=classical_message,
        message_dict=message_dict,
        feature_index=0,
        save_path=figures_dir / "message_passing_diff_feature_0.png",
        show=True,
    )

    # Feature 1 = q_mvar
    plot_message_passing_line_comparison(
        graph_input=graph_input,
        message_dict=message_dict,
        feature_index=1,
        save_path=figures_dir / "message_passing_line_feature_1.png",
        show=True,
    )
    plot_message_passing_graph_comparison(
        G=G,
        graph_input=graph_input,
        message_dict=message_dict,
        feature_index=1,
        save_path=figures_dir / "message_passing_graph_feature_1.png",
        show=True,
    )
    plot_message_passing_difference_to_classical(
        G=G,
        graph_input=graph_input,
        classical_message=classical_message,
        message_dict=message_dict,
        feature_index=1,
        save_path=figures_dir / "message_passing_diff_feature_1.png",
        show=True,
    )

    # Feature 2 = is_slack
    plot_message_passing_line_comparison(
        graph_input=graph_input,
        message_dict=message_dict,
        feature_index=2,
        save_path=figures_dir / "message_passing_line_feature_2.png",
        show=True,
    )
    plot_message_passing_graph_comparison(
        G=G,
        graph_input=graph_input,
        message_dict=message_dict,
        feature_index=2,
        save_path=figures_dir / "message_passing_graph_feature_2.png",
        show=True,
    )
    plot_message_passing_difference_to_classical(
        G=G,
        graph_input=graph_input,
        classical_message=classical_message,
        message_dict=message_dict,
        feature_index=2,
        save_path=figures_dir / "message_passing_diff_feature_2.png",
        show=True,
    )

    # -------------------------------------------------------------------------
    # Final metadata and log
    # -------------------------------------------------------------------------
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "output_dir": str(output_dir),
        "num_train": num_train,
        "num_test": int(x_test.shape[0]),
        "in_features": in_features,
        "hidden_features": hidden_features,
        "out_features": out_features,
        "scenario_index_used_for_inference": scenario_index,
        "top_models_in_inference_plot": top_models,
    }
    save_json(metadata, tables_dir / "run_metadata.json")
    save_text("\n".join(log_lines), logs_dir / "run_log.txt")

    print_block("Saved outputs")
    print(f"Figures: {figures_dir}")
    print(f"Tables : {tables_dir}")
    print(f"Arrays : {arrays_dir}")
    print(f"Logs   : {logs_dir}")


if __name__ == "__main__":
    main()
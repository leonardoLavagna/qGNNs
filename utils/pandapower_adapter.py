# -----------------------------------------------------------------------------
# pandapower_adapter.py
#
# Utilities to convert a pandapower network into a graph representation
# compatible with message-passing and GraphInput-style workflows.
#
# Main responsibilities:
# - build a NetworkX graph from a pandapower net
# - extract node and edge features in a consistent order
# - package the result in a reusable bundle
# - convert the bundle into the user's GraphInput object
# - run optional message passing
# - display graph signals before/after propagation
#
# Supported graph elements:
# - buses -> graph nodes
# - lines -> graph edges
# - transformers -> graph edges
#
# Notes:
# - Edge features are stored as dictionaries keyed by (u, v).
# - The graph is undirected.
# - If multiple elements connect the same bus pair, this module uses nx.Graph,
#   so only one edge can exist between a node pair. For most simple networks
#   this is fine. If needed, this can later be extended to nx.MultiGraph.
# -----------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandapower as pp


@dataclass
class PandapowerGraphBundle:
    """Container for graph objects derived from a pandapower net.

    Args:
        graph: NetworkX graph built from the pandapower net.
        node_order: Ordered list of bus indices used for node features.
        node_features: Node feature matrix with shape (num_nodes, num_features).
        edge_features: Edge feature dictionary keyed by (u, v).
        node_feature_names: Names of node feature columns.
        edge_feature_names: Names of edge feature columns.
        node_labels: Labels for display, keyed by node index.
    """

    graph: nx.Graph
    node_order: list[int]
    node_features: np.ndarray
    edge_features: dict[tuple[int, int], np.ndarray]
    node_feature_names: list[str]
    edge_feature_names: list[str]
    node_labels: dict[int, str]


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------
def _safe_edge_key(u: int, v: int) -> tuple[int, int]:
    """Return a canonical undirected edge key.

    Args:
        u: First node.
        v: Second node.

    Returns:
        Canonical edge key (min(u, v), max(u, v)).
    """
    return (u, v) if u <= v else (v, u)


def _is_line_in_service(net, line_idx: int) -> bool:
    """Check whether a line is operational.

    Args:
        net: Pandapower network.
        line_idx: Line index.

    Returns:
        True if the line is considered active.
    """
    if "in_service" in net.line.columns:
        return bool(net.line.loc[line_idx, "in_service"])
    return True


def _is_trafo_in_service(net, trafo_idx: int) -> bool:
    """Check whether a transformer is operational.

    Args:
        net: Pandapower network.
        trafo_idx: Transformer index.

    Returns:
        True if the transformer is considered active.
    """
    if "in_service" in net.trafo.columns:
        return bool(net.trafo.loc[trafo_idx, "in_service"])
    return True


def _is_branch_switch_closed(net, bus: int, element: int, et: str) -> bool:
    """Check if branch switches associated with an element are closed.

    Args:
        net: Pandapower network.
        bus: Bus index.
        element: Branch element index.
        et: Element type, usually 'l' for line or 't' for trafo.

    Returns:
        True if no relevant open switch disconnects the branch at this bus.
    """
    if len(net.switch) == 0:
        return True

    if not {"bus", "element", "et", "closed"}.issubset(set(net.switch.columns)):
        return True

    matches = net.switch[
        (net.switch["bus"] == bus)
        & (net.switch["element"] == element)
        & (net.switch["et"] == et)
    ]

    if len(matches) == 0:
        return True

    return bool(matches["closed"].all())


def _is_line_connected(net, line_idx: int) -> bool:
    """Check whether a line is connected and active.

    Args:
        net: Pandapower network.
        line_idx: Line index.

    Returns:
        True if the line should be included in the graph.
    """
    if not _is_line_in_service(net, line_idx):
        return False

    from_bus = int(net.line.loc[line_idx, "from_bus"])
    to_bus = int(net.line.loc[line_idx, "to_bus"])

    return (
        _is_branch_switch_closed(net, from_bus, line_idx, "l")
        and _is_branch_switch_closed(net, to_bus, line_idx, "l")
    )


def _is_trafo_connected(net, trafo_idx: int) -> bool:
    """Check whether a transformer is connected and active.

    Args:
        net: Pandapower network.
        trafo_idx: Transformer index.

    Returns:
        True if the transformer should be included in the graph.
    """
    if not _is_trafo_in_service(net, trafo_idx):
        return False

    hv_bus = int(net.trafo.loc[trafo_idx, "hv_bus"])
    lv_bus = int(net.trafo.loc[trafo_idx, "lv_bus"])

    return (
        _is_branch_switch_closed(net, hv_bus, trafo_idx, "t")
        and _is_branch_switch_closed(net, lv_bus, trafo_idx, "t")
    )


def _aggregate_load_per_bus(net, field: str, node_order: list[int]) -> np.ndarray:
    """Aggregate a load field for each bus.

    Args:
        net: Pandapower network.
        field: Load field name, e.g. 'p_mw' or 'q_mvar'.
        node_order: Ordered list of bus indices.

    Returns:
        One value per bus in node_order.
    """
    bus_to_pos = {bus_idx: i for i, bus_idx in enumerate(node_order)}
    values = np.zeros(len(node_order), dtype=float)

    if len(net.load) == 0:
        return values

    if field not in net.load.columns:
        return values

    for _, row in net.load.iterrows():
        if "in_service" in row and not bool(row["in_service"]):
            continue
        bus_idx = int(row["bus"])
        if bus_idx in bus_to_pos:
            values[bus_to_pos[bus_idx]] += float(row[field])

    return values


def _aggregate_sgen_per_bus(net, field: str, node_order: list[int]) -> np.ndarray:
    """Aggregate a static generator field for each bus.

    Args:
        net: Pandapower network.
        field: sgen field name, e.g. 'p_mw' or 'q_mvar'.
        node_order: Ordered list of bus indices.

    Returns:
        One value per bus in node_order.
    """
    bus_to_pos = {bus_idx: i for i, bus_idx in enumerate(node_order)}
    values = np.zeros(len(node_order), dtype=float)

    if not hasattr(net, "sgen") or len(net.sgen) == 0:
        return values

    if field not in net.sgen.columns:
        return values

    for _, row in net.sgen.iterrows():
        if "in_service" in row and not bool(row["in_service"]):
            continue
        bus_idx = int(row["bus"])
        if bus_idx in bus_to_pos:
            values[bus_to_pos[bus_idx]] += float(row[field])

    return values


def _aggregate_gen_per_bus(net, field: str, node_order: list[int]) -> np.ndarray:
    """Aggregate a generator field for each bus.

    Args:
        net: Pandapower network.
        field: Generator field name, e.g. 'p_mw' or 'vm_pu'.
        node_order: Ordered list of bus indices.

    Returns:
        One value per bus in node_order.
    """
    bus_to_pos = {bus_idx: i for i, bus_idx in enumerate(node_order)}
    values = np.zeros(len(node_order), dtype=float)

    if not hasattr(net, "gen") or len(net.gen) == 0:
        return values

    if field not in net.gen.columns:
        return values

    for _, row in net.gen.iterrows():
        if "in_service" in row and not bool(row["in_service"]):
            continue
        bus_idx = int(row["bus"])
        if bus_idx in bus_to_pos:
            values[bus_to_pos[bus_idx]] += float(row[field])

    return values


def _get_bus_feature(net, spec: str, node_order: list[int]) -> np.ndarray:
    """Extract one node feature vector.

    Args:
        net: Pandapower network.
        spec: Node feature specification.
        node_order: Ordered list of bus indices.

    Returns:
        Feature vector of shape (num_nodes,).

    Raises:
        ValueError: If the feature spec is not supported.
    """
    if spec == "vn_kv":
        return net.bus.loc[node_order, "vn_kv"].to_numpy(dtype=float)

    if spec == "in_service":
        if "in_service" not in net.bus.columns:
            return np.ones(len(node_order), dtype=float)
        return net.bus.loc[node_order, "in_service"].astype(float).to_numpy()

    if spec == "load_p_mw":
        return _aggregate_load_per_bus(net, "p_mw", node_order)

    if spec == "load_q_mvar":
        return _aggregate_load_per_bus(net, "q_mvar", node_order)

    if spec == "sgen_p_mw":
        return _aggregate_sgen_per_bus(net, "p_mw", node_order)

    if spec == "sgen_q_mvar":
        return _aggregate_sgen_per_bus(net, "q_mvar", node_order)

    if spec == "gen_p_mw":
        return _aggregate_gen_per_bus(net, "p_mw", node_order)

    if spec == "gen_vm_pu":
        return _aggregate_gen_per_bus(net, "vm_pu", node_order)

    if spec == "is_ext_grid":
        values = np.zeros(len(node_order), dtype=float)
        if hasattr(net, "ext_grid") and len(net.ext_grid) > 0:
            bus_to_pos = {bus_idx: i for i, bus_idx in enumerate(node_order)}
            for _, row in net.ext_grid.iterrows():
                if "in_service" in row and not bool(row["in_service"]):
                    continue
                bus_idx = int(row["bus"])
                if bus_idx in bus_to_pos:
                    values[bus_to_pos[bus_idx]] = 1.0
        return values

    if spec == "vm_pu":
        if not hasattr(net, "res_bus") or "vm_pu" not in net.res_bus.columns:
            raise ValueError("Node feature 'vm_pu' requires power-flow results.")
        return net.res_bus.loc[node_order, "vm_pu"].to_numpy(dtype=float)

    if spec == "va_degree":
        if not hasattr(net, "res_bus") or "va_degree" not in net.res_bus.columns:
            raise ValueError("Node feature 'va_degree' requires power-flow results.")
        return net.res_bus.loc[node_order, "va_degree"].to_numpy(dtype=float)

    if spec == "p_mw":
        if not hasattr(net, "res_bus") or "p_mw" not in net.res_bus.columns:
            raise ValueError("Node feature 'p_mw' requires power-flow results.")
        return net.res_bus.loc[node_order, "p_mw"].to_numpy(dtype=float)

    if spec == "q_mvar":
        if not hasattr(net, "res_bus") or "q_mvar" not in net.res_bus.columns:
            raise ValueError("Node feature 'q_mvar' requires power-flow results.")
        return net.res_bus.loc[node_order, "q_mvar"].to_numpy(dtype=float)

    raise ValueError(f"Unsupported node feature spec: {spec}")


def _get_line_feature(net, line_idx: int, spec: str) -> float:
    """Extract one feature from a line element.

    Args:
        net: Pandapower network.
        line_idx: Line index.
        spec: Edge feature specification.

    Returns:
        Scalar feature value.

    Raises:
        ValueError: If the feature spec is not supported.
    """
    if spec in {"length_km", "r_ohm_per_km", "x_ohm_per_km", "c_nf_per_km", "max_i_ka"}:
        if spec not in net.line.columns:
            raise ValueError(f"Line feature '{spec}' not found in net.line.")
        return float(net.line.loc[line_idx, spec])

    if spec in {"loading_percent", "p_from_mw", "q_from_mvar", "i_from_ka", "pl_mw", "ql_mvar"}:
        if not hasattr(net, "res_line") or spec not in net.res_line.columns:
            raise ValueError(f"Line result feature '{spec}' requires power-flow results.")
        return float(net.res_line.loc[line_idx, spec])

    if spec == "element_type":
        return 0.0

    raise ValueError(f"Unsupported line edge feature spec: {spec}")


def _get_trafo_feature(net, trafo_idx: int, spec: str) -> float:
    """Extract one feature from a transformer element.

    Args:
        net: Pandapower network.
        trafo_idx: Transformer index.
        spec: Edge feature specification.

    Returns:
        Scalar feature value.

    Raises:
        ValueError: If the feature spec is not supported.
    """
    if spec in {"sn_mva", "vn_hv_kv", "vn_lv_kv", "vk_percent", "vkr_percent", "pfe_kw", "i0_percent"}:
        if spec not in net.trafo.columns:
            raise ValueError(f"Transformer feature '{spec}' not found in net.trafo.")
        return float(net.trafo.loc[trafo_idx, spec])

    if spec in {"loading_percent", "p_hv_mw", "q_hv_mvar", "pl_mw", "ql_mvar", "i_hv_ka"}:
        if not hasattr(net, "res_trafo") or spec not in net.res_trafo.columns:
            raise ValueError(f"Transformer result feature '{spec}' requires power-flow results.")
        return float(net.res_trafo.loc[trafo_idx, spec])

    if spec == "element_type":
        return 1.0

    raise ValueError(f"Unsupported trafo edge feature spec: {spec}")


# -----------------------------------------------------------------------------
# Graph construction
# -----------------------------------------------------------------------------
def build_nx_graph_from_pandapower(
    net,
    include_lines: bool = True,
    include_trafos: bool = True,
    respect_switches: bool = True,
) -> nx.Graph:
    """Build a NetworkX graph from a pandapower network.

    Args:
        net: Pandapower network.
        include_lines: Whether to include line elements as edges.
        include_trafos: Whether to include transformer elements as edges.
        respect_switches: Whether to remove branches disconnected by open switches.

    Returns:
        NetworkX graph with buses as nodes and branches as edges.
    """
    G = nx.Graph()

    for bus_idx in net.bus.index:
        G.add_node(
            int(bus_idx),
            name=str(net.bus.loc[bus_idx, "name"]),
        )

    if include_lines and hasattr(net, "line"):
        for line_idx in net.line.index:
            if respect_switches and not _is_line_connected(net, int(line_idx)):
                continue
            from_bus = int(net.line.loc[line_idx, "from_bus"])
            to_bus = int(net.line.loc[line_idx, "to_bus"])
            G.add_edge(
                from_bus,
                to_bus,
                element="line",
                element_idx=int(line_idx),
            )

    if include_trafos and hasattr(net, "trafo"):
        for trafo_idx in net.trafo.index:
            if respect_switches and not _is_trafo_connected(net, int(trafo_idx)):
                continue
            hv_bus = int(net.trafo.loc[trafo_idx, "hv_bus"])
            lv_bus = int(net.trafo.loc[trafo_idx, "lv_bus"])
            G.add_edge(
                hv_bus,
                lv_bus,
                element="trafo",
                element_idx=int(trafo_idx),
            )

    return G


# -----------------------------------------------------------------------------
# Bundle construction
# -----------------------------------------------------------------------------
def pandapower_to_graph_bundle(
    net,
    node_feature_specs: Iterable[str] | None = None,
    edge_feature_specs: Iterable[str] | None = None,
    run_powerflow: bool = True,
    include_lines: bool = True,
    include_trafos: bool = True,
    respect_switches: bool = True,
) -> PandapowerGraphBundle:
    """Convert a pandapower net into a graph bundle.

    Args:
        net: Pandapower network.
        node_feature_specs: Node features to extract.
        edge_feature_specs: Edge features to extract.
        run_powerflow: Whether to run power flow before extracting results.
        include_lines: Whether to include line edges.
        include_trafos: Whether to include transformer edges.
        respect_switches: Whether to exclude branches opened by switches.

    Returns:
        PandapowerGraphBundle containing graph, features, and labels.
    """
    if node_feature_specs is None:
        node_feature_specs = ["load_p_mw", "load_q_mvar", "vm_pu"]

    if edge_feature_specs is None:
        edge_feature_specs = ["element_type", "loading_percent"]

    if run_powerflow:
        pp.runpp(net)

    G = build_nx_graph_from_pandapower(
        net=net,
        include_lines=include_lines,
        include_trafos=include_trafos,
        respect_switches=respect_switches,
    )

    node_order = list(map(int, net.bus.index))
    node_labels = {bus_idx: str(net.bus.loc[bus_idx, "name"]) for bus_idx in node_order}

    node_columns = []
    node_feature_names = list(node_feature_specs)

    for spec in node_feature_specs:
        values = _get_bus_feature(net, spec, node_order)
        node_columns.append(np.asarray(values, dtype=float).reshape(-1, 1))

    node_features = (
        np.hstack(node_columns)
        if len(node_columns) > 0
        else np.zeros((len(node_order), 0), dtype=float)
    )

    edge_features: dict[tuple[int, int], np.ndarray] = {}
    edge_feature_names = list(edge_feature_specs)

    for u, v, data in G.edges(data=True):
        element = data["element"]
        element_idx = int(data["element_idx"])

        feat = []
        for spec in edge_feature_specs:
            if element == "line":
                value = _get_line_feature(net, element_idx, spec)
            elif element == "trafo":
                value = _get_trafo_feature(net, element_idx, spec)
            else:
                raise ValueError(f"Unsupported edge element type: {element}")
            feat.append(value)

        edge_features[_safe_edge_key(u, v)] = np.asarray(feat, dtype=float)

    return PandapowerGraphBundle(
        graph=G,
        node_order=node_order,
        node_features=node_features,
        edge_features=edge_features,
        node_feature_names=node_feature_names,
        edge_feature_names=edge_feature_names,
        node_labels=node_labels,
    )


# -----------------------------------------------------------------------------
# Compatibility layer
# -----------------------------------------------------------------------------
def bundle_to_graph_input(
    bundle: PandapowerGraphBundle,
    build_graph_input_fn: Callable,
):
    """Convert a graph bundle into the user's GraphInput object.

    Args:
        bundle: Pandapower graph bundle.
        build_graph_input_fn: User function such as build_graph_input.

    Returns:
        GraphInput-like object produced by build_graph_input_fn.
    """
    return build_graph_input_fn(
        bundle.graph,
        node_signals=bundle.node_features,
        edge_signals=bundle.edge_features,
    )


def pandapower_to_graph_input(
    net,
    build_graph_input_fn: Callable,
    node_feature_specs: Iterable[str] | None = None,
    edge_feature_specs: Iterable[str] | None = None,
    run_powerflow: bool = True,
    include_lines: bool = True,
    include_trafos: bool = True,
    respect_switches: bool = True,
):
    """Convert a pandapower net directly into GraphInput.

    Args:
        net: Pandapower network.
        build_graph_input_fn: User function such as build_graph_input.
        node_feature_specs: Node features to extract.
        edge_feature_specs: Edge features to extract.
        run_powerflow: Whether to run power flow before extraction.
        include_lines: Whether to include line edges.
        include_trafos: Whether to include transformer edges.
        respect_switches: Whether to exclude branches opened by switches.

    Returns:
        GraphInput-like object produced by build_graph_input_fn.
    """
    bundle = pandapower_to_graph_bundle(
        net=net,
        node_feature_specs=node_feature_specs,
        edge_feature_specs=edge_feature_specs,
        run_powerflow=run_powerflow,
        include_lines=include_lines,
        include_trafos=include_trafos,
        respect_switches=respect_switches,
    )
    return bundle_to_graph_input(bundle, build_graph_input_fn)


# -----------------------------------------------------------------------------
# Message passing
# -----------------------------------------------------------------------------
def run_message_passing(graph_input, layer) -> np.ndarray:
    """Run one message-passing layer on a GraphInput object.

    Args:
        graph_input: User GraphInput object.
        layer: Layer exposing a forward(graph_input) method.

    Returns:
        Layer output as a NumPy array.
    """
    output = layer.forward(graph_input)
    return np.asarray(output, dtype=float)


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def _compute_graph_layout(
    G: nx.Graph,
    node_order: list[int],
    layout: str = "spring",
) -> dict[int, tuple[float, float]]:
    """Compute node positions for plotting.

    Args:
        G: Graph to plot.
        node_order: Ordered node list.
        layout: Layout name.

    Returns:
        Position dictionary keyed by node.

    Raises:
        ValueError: If the layout name is unsupported.
    """
    if layout == "spring":
        return nx.spring_layout(G, seed=42)

    if layout == "line":
        return {node: (i, 0.0) for i, node in enumerate(node_order)}

    if layout == "kamada_kawai":
        return nx.kamada_kawai_layout(G)

    if layout == "circular":
        return nx.circular_layout(G)

    raise ValueError(
        "Unsupported layout. Use one of: "
        "'spring', 'line', 'kamada_kawai', 'circular'."
    )


def _edge_values_from_bundle(
    bundle: PandapowerGraphBundle,
    edge_feature_index: int = 0,
) -> np.ndarray:
    """Extract one edge feature column in graph edge order.

    Args:
        bundle: Pandapower graph bundle.
        edge_feature_index: Column index of the edge feature to extract.

    Returns:
        Array of edge values in the same order as bundle.graph.edges().
    """
    values = []
    for u, v in bundle.graph.edges():
        key = _safe_edge_key(u, v)
        values.append(float(bundle.edge_features[key][edge_feature_index]))
    return np.asarray(values, dtype=float)


def display_graph_signal_comparison(
    G: nx.Graph,
    node_order: list[int],
    input_node_values: np.ndarray,
    output_node_values: np.ndarray | None = None,
    edge_values: np.ndarray | None = None,
    node_labels: Mapping[int, str] | None = None,
    layout: str = "spring",
    input_title: str = "Input node signal",
    output_title: str = "Output node signal",
    edge_title: str = "Edge signal",
    figsize: tuple[float, float] = (15.0, 4.0),
) -> None:
    """Display input/output node signals and an edge signal.

    Args:
        G: Graph to display.
        node_order: Ordered node list.
        input_node_values: Node values before message passing.
        output_node_values: Optional node values after message passing.
        edge_values: Optional edge values.
        node_labels: Optional labels for nodes.
        layout: Plot layout name.
        input_title: Title for the input panel.
        output_title: Title for the output panel.
        edge_title: Title for the edge panel.
        figsize: Figure size.

    Returns:
        None.
    """
    pos = _compute_graph_layout(G, node_order=node_order, layout=layout)
    labels = node_labels if node_labels is not None else {n: str(n) for n in G.nodes()}

    ncols = 1
    if output_node_values is not None:
        ncols += 1
    if edge_values is not None:
        ncols += 1

    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    if ncols == 1:
        axes = [axes]
    else:
        axes = list(axes)

    panel_idx = 0

    node_draw_0 = nx.draw_networkx_nodes(
        G,
        pos,
        node_color=np.asarray(input_node_values, dtype=float),
        cmap="viridis",
        node_size=900,
        ax=axes[panel_idx],
    )
    nx.draw_networkx_edges(G, pos, ax=axes[panel_idx])
    nx.draw_networkx_labels(G, pos, labels=labels, ax=axes[panel_idx])
    axes[panel_idx].set_title(input_title)
    axes[panel_idx].set_axis_off()
    plt.colorbar(node_draw_0, ax=axes[panel_idx], fraction=0.046, pad=0.04)
    panel_idx += 1

    if output_node_values is not None:
        node_draw_1 = nx.draw_networkx_nodes(
            G,
            pos,
            node_color=np.asarray(output_node_values, dtype=float),
            cmap="viridis",
            node_size=900,
            ax=axes[panel_idx],
        )
        nx.draw_networkx_edges(G, pos, ax=axes[panel_idx])
        nx.draw_networkx_labels(G, pos, labels=labels, ax=axes[panel_idx])
        axes[panel_idx].set_title(output_title)
        axes[panel_idx].set_axis_off()
        plt.colorbar(node_draw_1, ax=axes[panel_idx], fraction=0.046, pad=0.04)
        panel_idx += 1

    if edge_values is not None:
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color="lightgray",
            node_size=900,
            ax=axes[panel_idx],
        )
        nx.draw_networkx_edges(
            G,
            pos,
            edge_color=np.asarray(edge_values, dtype=float),
            edge_cmap=plt.cm.plasma,
            width=3.0,
            ax=axes[panel_idx],
        )
        nx.draw_networkx_labels(G, pos, labels=labels, ax=axes[panel_idx])
        axes[panel_idx].set_title(edge_title)
        axes[panel_idx].set_axis_off()

        edge_values = np.asarray(edge_values, dtype=float)
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.plasma,
            norm=plt.Normalize(
                vmin=float(np.min(edge_values)),
                vmax=float(np.max(edge_values)),
            ),
        )
        sm.set_array([])
        plt.colorbar(sm, ax=axes[panel_idx], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def display_pandapower_bundle(
    bundle: PandapowerGraphBundle,
    input_node_feature_index: int = 0,
    output_node_values: np.ndarray | None = None,
    edge_feature_index: int = 0,
    layout: str = "spring",
    figsize: tuple[float, float] = (15.0, 4.0),
) -> None:
    """Display graph signals stored in a graph bundle.

    Args:
        bundle: Pandapower graph bundle.
        input_node_feature_index: Index of the node feature to show as input.
        output_node_values: Optional propagated node values.
        edge_feature_index: Index of the edge feature to show.
        layout: Plot layout name.
        figsize: Figure size.

    Returns:
        None.
    """
    input_values = bundle.node_features[:, input_node_feature_index]
    edge_values = _edge_values_from_bundle(bundle, edge_feature_index=edge_feature_index)

    input_title = f"Input: {bundle.node_feature_names[input_node_feature_index]}"
    edge_title = f"Edges: {bundle.edge_feature_names[edge_feature_index]}"

    display_graph_signal_comparison(
        G=bundle.graph,
        node_order=bundle.node_order,
        input_node_values=input_values,
        output_node_values=output_node_values,
        edge_values=edge_values,
        node_labels=bundle.node_labels,
        layout=layout,
        input_title=input_title,
        output_title="After message passing",
        edge_title=edge_title,
        figsize=figsize,
    )


def display_pandapower_message_passing(
    net,
    build_graph_input_fn: Callable,
    layer=None,
    node_feature_specs: Iterable[str] | None = None,
    edge_feature_specs: Iterable[str] | None = None,
    input_node_feature_index: int = 0,
    edge_feature_index: int = 0,
    run_powerflow: bool = True,
    include_lines: bool = True,
    include_trafos: bool = True,
    respect_switches: bool = True,
    layout: str = "spring",
    figsize: tuple[float, float] = (15.0, 4.0),
):
    """Convert, optionally propagate, and display a pandapower net.

    Args:
        net: Pandapower network.
        build_graph_input_fn: User function such as build_graph_input.
        layer: Optional message-passing layer.
        node_feature_specs: Node features to extract.
        edge_feature_specs: Edge features to extract.
        input_node_feature_index: Node feature column to display as input.
        edge_feature_index: Edge feature column to display.
        run_powerflow: Whether to run power flow before extraction.
        include_lines: Whether to include line edges.
        include_trafos: Whether to include transformer edges.
        respect_switches: Whether to exclude branches opened by switches.
        layout: Plot layout name.
        figsize: Figure size.

    Returns:
        GraphInput-like object compatible with the user's code.
    """
    bundle = pandapower_to_graph_bundle(
        net=net,
        node_feature_specs=node_feature_specs,
        edge_feature_specs=edge_feature_specs,
        run_powerflow=run_powerflow,
        include_lines=include_lines,
        include_trafos=include_trafos,
        respect_switches=respect_switches,
    )

    graph_input = bundle_to_graph_input(bundle, build_graph_input_fn)

    output_values = None
    if layer is not None:
        output = run_message_passing(graph_input, layer)
        output = np.asarray(output, dtype=float)
        if output.ndim == 2:
            output_values = output[:, 0]
        else:
            output_values = output.reshape(-1)

    display_pandapower_bundle(
        bundle=bundle,
        input_node_feature_index=input_node_feature_index,
        output_node_values=output_values,
        edge_feature_index=edge_feature_index,
        layout=layout,
        figsize=figsize,
    )

    return graph_input


# -----------------------------------------------------------------------------
# Public feature reference
# -----------------------------------------------------------------------------
SUPPORTED_NODE_FEATURES = [
    "vn_kv",
    "in_service",
    "load_p_mw",
    "load_q_mvar",
    "sgen_p_mw",
    "sgen_q_mvar",
    "gen_p_mw",
    "gen_vm_pu",
    "is_ext_grid",
    "vm_pu",
    "va_degree",
    "p_mw",
    "q_mvar",
]

SUPPORTED_EDGE_FEATURES_LINES = [
    "element_type",
    "length_km",
    "r_ohm_per_km",
    "x_ohm_per_km",
    "c_nf_per_km",
    "max_i_ka",
    "loading_percent",
    "p_from_mw",
    "q_from_mvar",
    "i_from_ka",
    "pl_mw",
    "ql_mvar",
]

SUPPORTED_EDGE_FEATURES_TRAFOS = [
    "element_type",
    "sn_mva",
    "vn_hv_kv",
    "vn_lv_kv",
    "vk_percent",
    "vkr_percent",
    "pfe_kw",
    "i0_percent",
    "loading_percent",
    "p_hv_mw",
    "q_hv_mvar",
    "i_hv_ka",
    "pl_mw",
    "ql_mvar",
]
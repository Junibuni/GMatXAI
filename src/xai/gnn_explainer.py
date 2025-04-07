import os

import torch
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
from torch_geometric.utils import to_networkx

from src.utils.atom_info import ATOM_COLOR_MAP

def explain_graph_prediction(
    model,
    data,
    save_path=None,
    epochs=100,
    device="cpu",
    edge_threshold=0.3,
    scale_node_size=True
):
    model = model.to(device)
    data = data.to(device)
    model.eval()

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type='model',
        model_config=ModelConfig(
            mode='regression',
            task_level='graph',
            return_type='raw'  # model(data) → raw tensor
        ),
        edge_mask_type='object'
    )

    explanation = explainer(
        x=data.x,
        edge_index=data.edge_index,
    )

    edge_mask = explanation.edge_mask.detach().cpu()
    edge_mask = edge_mask.tolist()

    graph = to_networkx(data, to_undirected=True)
    pos = nx.spring_layout(graph, seed=42)
    
    edge_mask_min = min(edge_mask)
    edge_mask_max = max(edge_mask)

    filtered_edges = []
    edge_weights = []
    edge_widths = []
    for i, (u, v) in enumerate(graph.edges()):
        w = edge_mask[i]
        if w >= edge_threshold:
            filtered_edges.append((u, v))
            edge_weights.append(w)
            edge_widths.append(
                1 + 5 * ((w - edge_mask_min) / (edge_mask_max - edge_mask_min))
            )

    labels = {
        i: data.atom_types[i] if hasattr(data, "atom_types") else str(i)
        for i in range(data.num_nodes)
    }

    node_sizes = None
    if scale_node_size:
        degree_dict = dict(graph.degree())
        deg_values = [degree_dict[n] for n in graph.nodes()]
        deg_min = min(deg_values)
        deg_max = max(deg_values)
        node_sizes = [
            100 + 500 * ((degree_dict[n] - deg_min) / (deg_max - deg_min))
            for n in graph.nodes()
        ]
    else:
        node_sizes = [500 for _ in graph.nodes()]

    node_colors = [
        ATOM_COLOR_MAP.get(labels[i].rstrip('0123456789+-'), "#FA1691")
        for i in range(data.num_nodes)
    ]
    
    plt.figure(figsize=(8, 6))
    nx.draw(
        graph,
        pos,
        edgelist=filtered_edges,
        edge_color=edge_weights,
        edge_cmap=plt.cm.Blues,
        width=edge_widths,
        with_labels=True,
        labels=labels,
        node_color=node_colors,
        node_size=node_sizes,
        font_size=10,
        font_color="black"
    )

    plt.title("GNNExplainer - Edge Importance")
    
    unique_atoms = sorted(set([str(a).rstrip('0123456789+-') for a in data.atom_types]))
    legend_patches = [
        Patch(color=ATOM_COLOR_MAP.get(sym, "#FA1691"), label=sym)
        for sym in unique_atoms
    ]
    plt.legend(
        handles=legend_patches,
        title="Atom Types",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(unique_atoms)
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Explanation saved to: {save_path}")
    else:
        plt.show()

    return explanation

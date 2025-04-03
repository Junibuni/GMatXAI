# src/xai/gnn_explainer.py

import os
import torch
import matplotlib.pyplot as plt
import networkx as nx

from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
from torch_geometric.utils import to_networkx


def explain_graph_prediction(
    model,
    data,
    save_path=None,
    epochs=100,
    device="cpu"
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

    edge_mask = explanation.edge_mask.detach().cpu().tolist()
    graph = to_networkx(data, to_undirected=True)
    pos = nx.spring_layout(graph, seed=42)

    plt.figure(figsize=(8, 6))
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="lightblue",
        edge_color=edge_mask,
        edge_cmap=plt.cm.Blues,
        width=2,
    )
    plt.title("GNNExplainer - Edge Importance")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"✅ Explanation saved to: {save_path}")
    else:
        plt.show()

    return explanation

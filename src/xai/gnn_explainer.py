import os
import torch
import matplotlib.pyplot as plt

from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelConfig, ExplanationType, MaskType, ModelMode
from torch_geometric.utils import to_networkx
import networkx as nx


def explain_prediction(model, data, target_index=0, epochs=100, save_path=None, device="cpu"):
    model.eval()
    model.to(device)
    data = data.to(device)

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type=ExplanationType.model,
        model_config=ModelConfig(
            mode=ModelMode.regression,
            task_level="graph",
            return_type="raw"
        ),
        node_mask_type=MaskType.attributes,
        edge_mask_type=MaskType.object
    )

    explanation = explainer(
        x=data.x,
        edge_index=data.edge_index,
        index=None
    )

    edge_mask = explanation.edge_mask.detach().cpu()
    graph = to_networkx(data, to_undirected=True)
    pos = nx.spring_layout(graph)

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="lightblue",
        edge_color=edge_mask,
        edge_cmap=plt.cm.Blues,
        width=2,
        ax=ax
    )

    plt.title("GNNExplainer: Edge Importance")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Explanation saved to: {save_path}")
    else:
        plt.show()

    return explanation

def build_explainer(model, epochs=100):
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type=ExplanationType.model,
        model_config=ModelConfig(
            mode=ModelMode.regression,
            task_level="graph",
            return_type="raw"
        ),
        node_mask_type=MaskType.attributes,
        edge_mask_type=MaskType.object
    )
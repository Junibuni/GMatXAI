# src/xai/gnn_explainer.py

import os
import torch
import matplotlib.pyplot as plt
import networkx as nx

from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
from torch_geometric.utils import to_networkx

ATOM_COLOR_MAP = {
    'H': '#FFFFFF', 'He': '#D9FFFF', 'Li': '#CC80FF', 'Be': '#C2FF00', 'B': '#FFB5B5',
    'C': '#909090', 'N': '#3050F8', 'O': '#FF0D0D', 'F': '#90E050', 'Ne': '#B3E3F5',
    'Na': '#AB5CF2', 'Mg': '#8AFF00', 'Al': '#BFA6A6', 'Si': '#F0C8A0', 'P': '#FF8000',
    'S': '#FFFF30', 'Cl': '#1FF01F', 'Ar': '#80D1E3', 'K': '#8F40D4', 'Ca': '#3DFF00',
    'Sc': '#E6E6E6', 'Ti': '#BFC2C7', 'V': '#A6A6AB', 'Cr': '#8A99C7', 'Mn': '#9C7AC7',
    'Fe': '#E06633', 'Co': '#F090A0', 'Ni': '#50D050', 'Cu': '#C88033', 'Zn': '#7D80B0',
    'Ga': '#C28F8F', 'Ge': '#668080', 'As': '#BD80E3', 'Se': '#FFA100', 'Br': '#A62929',
    'Kr': '#5CB8D1', 'Rb': '#702EB0', 'Sr': '#00FF00', 'Y': '#94FFFF', 'Zr': '#94E0E0',
    'Nb': '#73C2C9', 'Mo': '#54B5B5', 'Tc': '#3B9E9E', 'Ru': '#248F8F', 'Rh': '#0A7D8C',
    'Pd': '#006985', 'Ag': '#C0C0C0', 'Cd': '#FFD98F', 'In': '#A67573', 'Sn': '#668080',
    'Sb': '#9E63B5', 'Te': '#D47A00', 'I': '#940094', 'Xe': '#429EB0', 'Cs': '#57178F',
    'Ba': '#00C900', 'La': '#70D4FF', 'Ce': '#FFFFC7', 'Pr': '#D9FFC7', 'Nd': '#C7FFC7',
    'Pm': '#A3FFC7', 'Sm': '#8FFFC7', 'Eu': '#61FFC7', 'Gd': '#45FFC7', 'Tb': '#30FFC7',
    'Dy': '#1FFFC7', 'Ho': '#00FF9C', 'Er': '#00E675', 'Tm': '#00D452', 'Yb': '#00BF38',
    'Lu': '#00AB24', 'Hf': '#4DC2FF', 'Ta': '#4DA6FF', 'W': '#2194D6', 'Re': '#267DAB',
    'Os': '#266696', 'Ir': '#175487', 'Pt': '#D0D0E0', 'Au': '#FFD123', 'Hg': '#B8B8D0',
    'Tl': '#A6544D', 'Pb': '#575961', 'Bi': '#9E4FB5', 'Po': '#AB5C00', 'At': '#754F45',
    'Rn': '#428296', 'Fr': '#420066', 'Ra': '#007D00', 'Ac': '#70ABFA', 'Th': '#00BAFF',
    'Pa': '#00A1FF', 'U': '#008FFF', 'Np': '#0080FF', 'Pu': '#006BFF', 'Am': '#545CF2',
    'Cm': '#785CE3', 'Bk': '#8A4FE3', 'Cf': '#A136D4', 'Es': '#B31FD4', 'Fm': '#B31FBA',
    'Md': '#B30DA6', 'No': '#BD0D87', 'Lr': '#C70066', 'Rf': '#CC0059', 'Db': '#D1004F',
    'Sg': '#D90045', 'Bh': '#E00038', 'Hs': '#E6002E', 'Mt': '#EB0026', 'Ds': '#FF1493',
    'Rg': '#FF00AF', 'Cn': '#FF007F', 'Fl': '#FF0055', 'Lv': '#FF0022', 'Ts': '#FF0000',
    'Og': '#E00000'
}

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

    filtered_edges = [
        (u, v) for i, (u, v) in enumerate(graph.edges())
        if edge_mask[i] >= edge_threshold
    ]
    edge_weights = [
        edge_mask[i] for i, (u, v) in enumerate(graph.edges())
        if edge_mask[i] >= edge_threshold
    ]

    labels = {
        i: data.atom_types[i] if hasattr(data, "atom_types") else str(i)
        for i in range(data.num_nodes)
    }

    node_sizes = None
    if scale_node_size:
        degree_dict = dict(graph.degree())
        node_sizes = [300 + 50 * degree_dict[n] for n in graph.nodes()]

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
        width=2,
        with_labels=True,
        labels=labels,
        node_color=node_colors,
        node_size=node_sizes,
        font_size=10,
        font_color="black"
    )

    plt.title("GNNExplainer - Edge Importance")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Explanation saved to: {save_path}")
    else:
        plt.show()

    return explanation

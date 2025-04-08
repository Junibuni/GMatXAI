import os
import json

import torch
from torch_geometric.explain import Explainer, ModelConfig
from torch_geometric.explain.algorithm import (GNNExplainer,
                                               AttentionExplainer,
                                               CaptumExplainer,
                                               PGExplainer)

from src.xai.gnn_explainer import explain_graph_prediction

# https://pytorch-geometric.readthedocs.io/en/latest/modules/explain.html#philosophy
def get_explainer(model, algorithm_name="gnn_explainer", epochs=100):
    if algorithm_name == "gnn_explainer":
        algorithm = GNNExplainer(epochs=epochs)
    # TODO
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")

    return Explainer(
        model=model,
        algorithm=algorithm,
        explanation_type='model',
        model_config=ModelConfig(
            mode='regression',
            task_level='graph',
            return_type='raw'
        ),
        edge_mask_type='object'
    )


def explain_batch(
    model,
    data_list,
    save_dir,
    device="cpu",
    epochs=100,
    edge_threshold=0.3,
    scale_node_size=True,
    algorithm="gnn_explainer"
):
    os.makedirs(save_dir, exist_ok=True)
    json_dir = os.path.join(save_dir, "json")
    os.makedirs(json_dir, exist_ok=True)
    error_log_path = os.path.join(save_dir, "errors.log")

    model = model.to(device)
    model.eval()
    explainer = get_explainer(model, algorithm_name=algorithm, epochs=epochs)

    with open(error_log_path, "w") as err_log:
        for i, data in enumerate(data_list):
            try:
                data = data.to(device)
                material_id = getattr(data, "material_id", f"sample_{i}")
                save_path = os.path.join(save_dir, f"{material_id}.png")

                explanation = explain_graph_prediction(
                    model=model,
                    data=data,
                    save_path=save_path,
                    epochs=epochs,
                    device=device,
                    edge_threshold=edge_threshold,
                    scale_node_size=scale_node_size,
                    explainer=explainer
                )

                explanation_dict = {
                    'edge_mask': explanation.edge_mask.tolist() if explanation.edge_mask is not None else None,
                    'prediction': explanation.prediction.tolist() if explanation.prediction is not None else None,
                    'target': explanation.target.tolist() if explanation.target is not None else None,
                    'x': explanation.x.tolist() if explanation.x is not None else None,
                    'edge_index': explanation.edge_index.tolist() if explanation.edge_index is not None else None
                }

                json_path = os.path.join(json_dir, f"{material_id}.json")
                with open(json_path, 'w') as f:
                    json.dump(explanation_dict, f, indent=4)

            except Exception as e:
                err_msg = f"Failed to explain {getattr(data, 'material_id', f'#{i}')}: {str(e)}\n"
                print(err_msg.strip())
                err_log.write(err_msg)
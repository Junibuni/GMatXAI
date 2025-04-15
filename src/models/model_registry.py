from src.models.cgcnn import CGCNN
from src.models.cartnet import CartNet
# from src.models.megnet import MEGNet

MODEL_REGISTRY = {
    "cgcnn": CGCNN,
    "cartnet": CartNet
    # "megnet": MEGNet,
    # "alignn": ALIGNN,
}

def get_model(name, config=None):
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {name}")

    model_class = MODEL_REGISTRY[name]

    if config is not None:
        return model_class.from_config(config)
    return model_class
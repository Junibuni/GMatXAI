import os
import torch


def save_model(state_dict, save_dir, model_name="cgcnn", epoch=None):
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{model_name}_best.pth" if epoch is None else f"{model_name}_epoch{epoch}.pth"
    save_path = os.path.join(save_dir, filename)

    torch.save(state_dict, save_path)
    print(f"Model saved to: {save_path}")


def load_model(model, path, device="cpu"):
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Model loaded from: {path}")
    return model

"""
Example

save_model(best_model_state, save_dir="checkpoints", model_name="cgcnn")

model = CGCNN(...)
model = load_model(model, "checkpoints/cgcnn_best.pth", device="cuda")
"""
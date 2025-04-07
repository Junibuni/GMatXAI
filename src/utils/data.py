import random

def sample_explanation_data(test_dataset, material_ids=None, k=3):
    if material_ids:
        selected = [d for d in test_dataset if getattr(d, "material_id", None) in material_ids]
        remaining = [d for d in test_dataset if d not in selected]
        needed = max(0, k - len(selected))

        if needed > 0 and len(remaining) >= needed:
            selected += random.sample(remaining, k=needed)
    else:
        selected = random.sample(test_dataset, k=min(k, len(test_dataset)))

    return selected
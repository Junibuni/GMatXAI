import random

def sample_explanation_data(test_dataset, material_ids=None, k=3):
    if material_ids:
        selected = [d for d in test_dataset if getattr(d, "material_id", None) in material_ids]
        remaining = [d for d in test_dataset if d not in selected]
        needed = max(0, k - len(selected))

        if needed > 0 and len(remaining) >= needed:
            selected += random.sample(remaining, k=needed)
    else:
        indices = random.sample(range(len(test_dataset)), k=min(k, len(test_dataset)))
        selected = [test_dataset[i] for i in indices]

    return selected

def standardize(data, mean, std):
    return (data - mean) / std

def reverse_standardization(standardized_data, mean, std):
    return standardized_data * std + mean
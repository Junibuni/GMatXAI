import torch

def get_optimizer(optim_type: str, model_parameters, lr, **kwargs):
    optim_type = optim_type.lower()

    if optim_type == "adam":
        weight_decay = kwargs.get("weight_decay", 0.0)
        return torch.optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)

    elif optim_type == "sgd":
        momentum = kwargs.get("momentum", 0.9)
        weight_decay = kwargs.get("weight_decay", 0.0)

        return torch.optim.SGD(
            model_parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )

    elif optim_type == "adamw":
        weight_decay = kwargs.get("weight_decay", 0.01)
        return torch.optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)

    else:
        raise ValueError(f"Unsupported optimizer: {optim_type}")

def get_scheduler(scheduler_type: str, optimizer, **kwargs):
    scheduler_type = scheduler_type.lower()

    if scheduler_type == "step":
        step_size = kwargs.get("step_size")
        if step_size == 0:
            step_size = kwargs.get("steps_per_epoch")
            
        gamma = kwargs.get("gamma", 0.1)

        if step_size is None:
            raise ValueError("StepLR requires 'step_size' in kwargs")

        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )

    elif scheduler_type == "cosine":
        T_max = kwargs.get("T_max")

        if T_max is None:
            raise ValueError("CosineAnnealingLR requires 'T_max' in kwargs")

        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    
    elif scheduler_type == "onecycle":
        max_lr = kwargs.get("max_lr")
        total_steps = kwargs.get("total_steps")
        epochs = kwargs.get("epochs")
        steps_per_epoch = kwargs.get("steps_per_epoch")
        pct_start = kwargs.get("pct_start", 0.3)
        anneal_strategy = kwargs.get("anneal_strategy", "cos")
        div_factor = kwargs.get("div_factor", 25.0)
        final_div_factor = kwargs.get("final_div_factor", 1e4)

        if max_lr is None:
            raise ValueError("OneCycleLR requires 'max_lr' in kwargs")

        if total_steps is None:
            if epochs is None or steps_per_epoch is None:
                raise ValueError("OneCycleLR requires either 'total_steps' or both 'epochs' and 'steps_per_epoch'")
            total_steps = epochs * steps_per_epoch

        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            div_factor=div_factor,
            final_div_factor=final_div_factor
        )

    elif scheduler_type == "none" or scheduler_type is None:
        return None

    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_type}")

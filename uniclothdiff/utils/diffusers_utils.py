
def build_diffusion_scheduler(name, config):
    name = f"{name}Scheduler"
    try:
        scheduler_cls = getattr(__import__('diffusers', fromlist=[name]), name)
    except AttributeError:
        raise ImportError(f"Scheduler class '{name}' not found in diffusers library.")

    return scheduler_cls(**config)
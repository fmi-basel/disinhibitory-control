import logging
import jax

# SET UP LOGGER
logging.basicConfig()
logger = logging.getLogger("incontrol.utils")


def select_device(device, gpu_id) -> None:
    """
    Selects device to run on.
    """
    if device == 'cpu':
        logger.info("Using CPU...")
        device = jax.devices('cpu')[0]
        jax.config.update('jax_platform_name', 'cpu')
        
    elif device == 'gpu':
        
        num_gpus = jax.devices('gpu')
        assert gpu_id < len(num_gpus), f"GPU {gpu_id} not available. Only {len(num_gpus)} GPUs available."

        logger.info(f"Using GPU {gpu_id}...")
        device = jax.devices('gpu')[gpu_id]
        jax.config.update('jax_platform_name', 'gpu')
        
    else:
        raise ValueError("Unknown device. Must be 'cpu' or 'gpu'.")
        
    jax.config.update("jax_default_device", device)
    return device



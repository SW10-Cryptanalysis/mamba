from dataclasses import asdict
import torch
from transformers import Mamba2Config, Mamba2ForCausalLM
from src.config import Config
from src.utils.logging import get_logger

logger = get_logger(__name__)

def get_model(config: Config) -> Mamba2ForCausalLM:
    """Initialize a Mamba2 model with parameters defined in the project configuration.

    Args:
        config (Config): The global project configuration object containing
            `mamba_config` (the architecture hyperparameters).

    Returns:
        Mamba2ForCausalLM: An initialized Mamba2 model ready for training or inference.

    """
    m_dict = asdict(config.mamba_config)

    config = Mamba2Config(
        **m_dict,
        torch_dtype=torch.bfloat16,
    )

    model = Mamba2ForCausalLM(config)

    logger.info("Mamba2 Model loaded!")
    logger.info(f"Parameters:       {model.num_parameters():,}")
    logger.info(f"VRAM for Weights: {(model.get_memory_footprint() / 1e9):.4f} GB")

    return model

from dataclasses import asdict
import torch
from transformers import Mamba2Config, Mamba2ForCausalLM
from src.config import Config
from src.utils.logging import get_logger

logger = get_logger(__name__)

def get_model(config: Config) -> Mamba2ForCausalLM:
    """Init Mamba2 model with params from config."""
    m_dict = asdict(config.mamba_config)

    config = Mamba2Config(
        **m_dict,
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )

    model = Mamba2ForCausalLM(config)

    logger.info("Mamba2 Model loaded!")
    logger.info(f"Parameters:       {model.num_parameters():,}")
    logger.info(f"VRAM for Weights: {(model.get_memory_footprint() / 1e9):.4f} GB")

    return model

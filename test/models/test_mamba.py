from dataclasses import dataclass
from typing import Any

import pytest
import torch

from src.models.mamba import get_model


@dataclass
class DummyMambaConfig:
    """Dummy dataclass to mimic src.config.MambaConfig for asdict extraction."""

    d_model: int
    n_layer: int
    vocab_size: int


@dataclass
class GetModelTestCase:
    """Data structure for testing Mamba model initialization and logging."""

    name: str
    d_model: int
    n_layer: int
    vocab_size: int
    mock_params: int
    mock_vram_bytes: int
    expected_vram_gb: str


test_cases = [
    GetModelTestCase(
        name="small_model",
        d_model=768,
        n_layer=24,
        vocab_size=50277,
        mock_params=130_000_000,
        mock_vram_bytes=260_000_000,
        expected_vram_gb="0.2600",
    ),
    GetModelTestCase(
        name="large_model",
        d_model=2048,
        n_layer=48,
        vocab_size=50277,
        mock_params=350_000_000,
        mock_vram_bytes=700_000_000,
        expected_vram_gb="0.7000",
    ),
]


@pytest.mark.parametrize("case", test_cases, ids=lambda c: c.name)
def test_get_model(case: GetModelTestCase, mocker: Any) -> None:
    """Tests the initialization, configuration, and logging of the Mamba2 model."""
    config = DummyMambaConfig(
        d_model=case.d_model, n_layer=case.n_layer, vocab_size=case.vocab_size
    )

    mock_mamba2_config_cls = mocker.patch("src.models.mamba.Mamba2Config")
    mock_mamba2_model_cls = mocker.patch("src.models.mamba.Mamba2ForCausalLM")
    mock_logger = mocker.patch("src.models.mamba.logger")

    mock_model_instance = mock_mamba2_model_cls.return_value
    mock_model_instance.num_parameters.return_value = case.mock_params
    mock_model_instance.get_memory_footprint.return_value = case.mock_vram_bytes

    result = get_model(config)  # type: ignore

    mock_mamba2_config_cls.assert_called_once_with(
        d_model=case.d_model, n_layer=case.n_layer, vocab_size=case.vocab_size
    )

    mamba2_config_instance = mock_mamba2_config_cls.return_value
    assert mamba2_config_instance.torch_dtype == torch.bfloat16

    mock_mamba2_model_cls.assert_called_once_with(mamba2_config_instance)
    assert result == mock_model_instance

    mock_logger.info.assert_any_call("Mamba2 Model loaded!")
    mock_logger.info.assert_any_call(f"Parameters:       {case.mock_params:,}")
    mock_logger.info.assert_any_call(f"VRAM for Weights: {case.expected_vram_gb} GB")

    assert mock_logger.info.call_count == 3

import pytest
import torch
import torch.nn as nn
from dataclasses import dataclass
from unittest.mock import MagicMock

from src.config import Config
from src.models.mamba import MambaModel


@dataclass
class MambaModelTestCase:
    """Dataclass for MambaModel test parameters."""

    name: str
    vocab_size: int
    char_offset: int
    batch_size: int
    seq_len: int
    use_inference_params: bool


@pytest.fixture
def mock_config():
    """Fixture providing a standard model configuration."""
    config = MagicMock(spec=Config)
    config.d_model = 128
    config.d_state = 16
    config.d_conv = 4
    config.expand = 2
    config.n_layers = 2
    return config


@pytest.mark.parametrize(
    "case",
    [
        MambaModelTestCase(
            name="standard_forward_pass",
            vocab_size=1000,
            char_offset=500,
            batch_size=2,
            seq_len=16,
            use_inference_params=False,
        ),
        MambaModelTestCase(
            name="incremental_inference_pass",
            vocab_size=1000,
            char_offset=500,
            batch_size=1,
            seq_len=1,
            use_inference_params=True,
        ),
    ],
    ids=lambda x: x.name,
)
def test_mamba_model_lifecycle(mocker, mock_config, case: MambaModelTestCase):
    """Verify initialization, layer structure, and forward pass flow."""

    mock_mixer = mocker.patch(
        "src.models.mamba.Mamba2", return_value=mocker.MagicMock(spec=nn.Module)
    )
    mock_norm = mocker.patch(
        "src.models.mamba.RMSNorm", return_value=mocker.MagicMock(spec=nn.Module)
    )

    def mock_mixer_forward(x, inference_params=None):
        """Simulate mixer output maintaining input shape."""
        return x

    def mock_norm_forward(x):
        """Simulate norm output maintaining input shape."""
        return x

    mock_mixer.return_value.side_effect = mock_mixer_forward
    mock_norm.return_value.side_effect = mock_norm_forward

    model = MambaModel(
        vocab_size=case.vocab_size, char_offset=case.char_offset, config=mock_config
    )

    assert model.char_offset == case.char_offset
    assert len(model.layers) == mock_config.n_layers
    assert isinstance(model.embedding, nn.Embedding)
    assert isinstance(model.lm_head, nn.Linear)

    input_ids = torch.randint(0, case.vocab_size, (case.batch_size, case.seq_len))
    inf_params = mocker.Mock() if case.use_inference_params else None

    logits = model.forward(input_ids, inference_params=inf_params)

    assert logits.shape == (case.batch_size, case.seq_len, case.vocab_size)

    assert mock_mixer.call_count == mock_config.n_layers
    for _ in range(mock_config.n_layers):
        mock_mixer.return_value.assert_any_call(mocker.ANY, inference_params=inf_params)

    assert mock_norm.call_count == mock_config.n_layers + 1

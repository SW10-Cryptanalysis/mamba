import pytest
import torch
from unittest.mock import MagicMock, patch
from transformers import Mamba2Config
from src.models.mamba import get_model

@pytest.fixture
def mock_config():
    """Mocks the project Config and the nested Mamba2Config dataclass."""
    config = MagicMock()

    mamba_params = MagicMock()
    mamba_params.vocab_size = 100
    mamba_params.d_model = 64
    mamba_params.n_layer = 2

    config.mamba_config = mamba_params
    return config

class TestMambaModel:

    @patch("src.models.mamba.asdict")
    @patch("src.models.mamba.Mamba2ForCausalLM")
    def test_get_model_initialization(self, mock_model_cls, mock_asdict, mock_config):
        """Test if the model is initialized with the correct config values."""
        fake_params = {"vocab_size": 128, "d_model": 256, "use_cache": False}
        mock_asdict.return_value = fake_params

        mock_instance = MagicMock()
        mock_instance.num_parameters.return_value = 1000000
        mock_instance.get_memory_footprint.return_value = 4000000
        mock_model_cls.return_value = mock_instance

        model = get_model(mock_config)

        mock_asdict.assert_called_once_with(mock_config.mamba_config)

        assert mock_model_cls.called

        called_config = mock_model_cls.call_args[0][0]
        assert isinstance(called_config, Mamba2Config)
        assert called_config.use_cache is False

        mock_instance.num_parameters.assert_called()
        assert model == mock_instance

    def test_model_dtype(self, mock_config):
        """A 'semi-integration' test to ensure config object has correct dtype."""
        with patch("src.models.mamba.Mamba2ForCausalLM") as mock_cls, \
             patch("src.models.mamba.asdict", return_value={"vocab_size": 10}):

            mock_instance = MagicMock()
            mock_instance.num_parameters.return_value = 1000
            mock_instance.get_memory_footprint.return_value = 4000
            mock_cls.return_value = mock_instance

            get_model(mock_config)

            actual_config = mock_cls.call_args[0][0]
            assert actual_config.torch_dtype == torch.bfloat16

import pytest
from unittest.mock import MagicMock, patch
from src.config import Config
from src.train import resolve_config, train_model

@pytest.fixture
def mock_exp_dir(tmp_path):
    """Create a fake experiment directory with a config.json."""
    exp_dir = tmp_path / "exp_0101_1200_2024"
    exp_dir.mkdir()
    config_file = exp_dir / "config.json"
    # Create a config with a specific value to see if it overrides
    config_file.write_text('{"learning_rate": 0.0099, "batch_size": 123}')

    checkpoint = exp_dir / "latest.pth"
    checkpoint.write_text("fake_weights")
    return checkpoint

def test_resolve_config_manual_path(mock_exp_dir):
    """Verify that providing a manual path loads the corresponding config."""
    config = Config()
    config.learning_rate = 0.001 # Default

    resume_path, target_dir = resolve_config(str(mock_exp_dir), config, "normal")

    assert resume_path == mock_exp_dir
    assert target_dir == mock_exp_dir.parent
    # Check if the override worked
    assert config.learning_rate == 0.0099
    assert config.batch_size == 123

@patch("src.utils.data_manager.DataManager.get_latest_checkpoint")
def test_resolve_config_auto(mock_get_latest, mock_exp_dir):
    """Verify that 'auto' correctly triggers the DataManager search."""
    mock_get_latest.return_value = mock_exp_dir
    config = Config()

    resume_path, _ = resolve_config("auto", config, "normal")

    mock_get_latest.assert_called_once()
    assert resume_path == mock_exp_dir

@patch("src.train.MambaTrainer")
@patch("src.train.MambaModel")
@patch("src.train.get_loaders")
@patch("src.train.CipherTokenizer")
def test_train_model_flow(mock_tok, mock_loaders, mock_model, mock_trainer_class):
    """Ensure the full pipeline orchestrates components in the right order."""

    # Setup mocks
    mock_loaders.return_value = (MagicMock(), MagicMock()) # (train, val)
    mock_trainer_instance = mock_trainer_class.return_value

    # Run the function
    train_model(resume_arg=None, device="cpu")

    # Assertions: Did it move to CUDA? Did it start training?
    mock_model.return_value.to.assert_called_with("cpu")
    mock_trainer_instance.train.assert_called_once()

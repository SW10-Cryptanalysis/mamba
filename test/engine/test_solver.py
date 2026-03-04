import pytest
import torch
from unittest.mock import MagicMock, patch
from src.engine.solver import CipherSolver
from src.config import Config

@pytest.fixture
def config():
    c = Config()
    c.max_len = 10
    return c

@pytest.fixture
def solver(config):
    return CipherSolver(config, device="cpu")

def test_calculate_ser(solver):
    """Test the Symbol Error Rate math."""
    assert pytest.approx(solver.calculate_ser("hello", "hella")) == 0.2
    assert solver.calculate_ser("abc", "abc") == 0.0
    assert solver.calculate_ser("abc", "xyz") == 1.0

@patch("torch.load")
@patch("src.engine.solver.MambaModel")
def test_load_checkpoint(mock_mamba_class, mock_torch_load, solver, tmp_path):
    """Verify that loading a checkpoint sets up the model and metadata correctly."""
    # 1. Mock the checkpoint dictionary
    mock_torch_load.return_value = {
        "char_offset": 100,
        "model_state_dict": {},
        "val_loss": 0.5
    }

    # 2. Create a dummy path
    ckpt_path = tmp_path / "test.pth"
    ckpt_path.write_text("dummy content")

    # 3. Call load
    solver.load_checkpoint(ckpt_path)

    # 4. Assertions
    assert solver.model is not None
    assert solver.metadata["val_loss"] == 0.5
    # Ensure the model was set to eval mode
    solver.model.eval.assert_called_once()

@patch("src.engine.solver.CipherSolver.load_checkpoint")
def test_decrypt_input_formats(mock_load, solver):
    """Verify decrypt handles both list[int] and space-separated strings."""
    # Mock the model and its return value
    solver.model = MagicMock()

    # Create fake logits
    fake_logits = torch.zeros((1, 3, 500))
    fake_logits[0, :, 102] = 10.0
    solver.model.return_value = fake_logits

    # Test String Input
    res1 = solver.decrypt("1 2 3")
    # Test List Input
    res2 = solver.decrypt([1, 2, 3])

    assert isinstance(res1, str)
    assert res1 == res2

def test_decrypt_before_load_raises_error(solver):
    """Ensure we can't decrypt without a model."""
    with pytest.raises(RuntimeError, match="Model not loaded"):
        solver.decrypt("1 2 3")

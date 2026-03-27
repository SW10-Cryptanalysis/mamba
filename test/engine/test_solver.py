import pytest
import torch
import json
from unittest.mock import MagicMock, patch
from src.engine.solver import MambaCipherSolver

@pytest.fixture
def mock_config():
    config = MagicMock()
    config.bos_token_id = 0
    config.sep_token_id = 1
    config.eos_token_id = 2
    config.pad_token_id = 3
    config.space_token_id = 4
    config.char_offset = 10
    config.use_spaces = False
    return config

@pytest.fixture
def solver(mock_config, tmp_path):
    """Creates a solver instance with a secure temp path and mocked weights."""
    model_dir = tmp_path / "fake_model"
    model_dir.mkdir()

    with patch("src.engine.solver.Mamba2ForCausalLM.from_pretrained") as mock_from_pretrained:
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.config = MagicMock()
        mock_from_pretrained.return_value = mock_model

        s = MambaCipherSolver(model_path=str(model_dir), config=mock_config)
        return s

class TestMambaCipherSolver:

    def test_decode_tokens_normal(self, solver, mock_config):
        """Test standard character decoding logic."""
        token_ids = [10, 11, 12]
        decoded = solver._decode_tokens(token_ids)
        assert decoded == "abc"

    def test_decode_tokens_with_spaces(self, solver, mock_config):
        """Test space decoding logic."""
        mock_config.use_spaces = True
        token_ids = [10, 4, 11]
        decoded = solver._decode_tokens(token_ids)
        assert decoded == "a_b"

    def test_calculate_ser_perfect(self, solver):
        """Test SER when prediction is perfect."""
        ser = solver._calculate_ser("hello", "hello")
        assert ser == 0.0

    def test_calculate_ser_wrong(self, solver):
        """Test SER with mismatches and length differences."""
        ser = solver._calculate_ser("hello", "jello!")
        assert ser == 0.4

    def test_solve(self, solver, mock_config):
        """Test the solve pipeline (tensor creation to decoding)."""
        cipher_ids = [50, 51]

        mock_output = torch.tensor([[0, 50, 51, 1, 10, 11]])
        solver.model.generate = MagicMock(return_value=mock_output)

        result = solver.solve(cipher_ids)

        assert solver.model.generate.called
        assert result == "ab"

    def test_evaluate(self, solver, tmp_path, mock_config):
        """Test the full evaluation loop and file logging."""
        solver.model_path = tmp_path

        mock_dataset = [
            {
                "input_ids": [0, 50, 51, 1, 10, 11],
                "raw_plaintext": "ab"
            }
        ]

        solver.solve = MagicMock(return_value="ab")

        solver.evaluate(mock_dataset)

        log_file = tmp_path / "evaluation_results.jsonl"
        assert log_file.exists()

        with open(log_file) as f:
            data = json.loads(f.readline())
            assert data["target"] == "ab"
            assert data["pred"] == "ab"
            assert data["ser"] == 0.0

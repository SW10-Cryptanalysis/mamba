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
def solver(mock_config):
    # Mocking from_pretrained so it doesn't try to download/load weights
    with patch("transformers.Mamba2ForCausalLM.from_pretrained") as mock_from_pretrained:
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_from_pretrained.return_value = mock_model

        # Instantiate solver
        s = MambaCipherSolver(model_path="/tmp/fake_model", config=mock_config)
        return s

class TestMambaCipherSolver:

    def test_decode_tokens_normal(self, solver, mock_config):
        """Test standard character decoding logic."""
        # offset 10 + (ord('a') - ord('a')) = 10 ('a')
        # offset 10 + (ord('b') - ord('a')) = 11 ('b')
        token_ids = [10, 11, 12] # a, b, c
        decoded = solver._decode_tokens(token_ids)
        assert decoded == "abc"

    def test_decode_tokens_with_spaces(self, solver, mock_config):
        """Test space decoding logic."""
        mock_config.use_spaces = True
        token_ids = [10, 4, 11] # a, space, b
        decoded = solver._decode_tokens(token_ids)
        assert decoded == "a_b"

    def test_calculate_ser_perfect(self, solver):
        """Test SER when prediction is perfect."""
        ser = solver._calculate_ser("hello", "hello")
        assert ser == 0.0

    def test_calculate_ser_wrong(self, solver):
        """Test SER with mismatches and length differences."""
        # 1 mismatch ('h' vs 'j') + 1 length diff ('!')
        # Total error = 2 / 5 = 0.4
        ser = solver._calculate_ser("hello", "jello!")
        assert ser == 0.4

    def test_solve(self, solver, mock_config):
        """Test the solve pipeline (tensor creation to decoding)."""
        cipher_ids = [50, 51]

        # Mock the model.generate output
        # Input: [BOS, 50, 51, SEP] (len 4)
        # We want to simulate the model returning [BOS, 50, 51, SEP, 10, 11]
        mock_output = torch.tensor([[0, 50, 51, 1, 10, 11]])
        solver.model.generate = MagicMock(return_value=mock_output)

        result = solver.solve(cipher_ids)

        # Verify generate was called
        assert solver.model.generate.called
        # Verify we only decoded the 'newly' generated tokens (10, 11 -> ab)
        assert result == "ab"

    def test_evaluate(self, solver, tmp_path, mock_config):
        """Test the full evaluation loop and file logging."""
        # Redirect log to a temp pytest directory
        solver.model_path = tmp_path

        mock_dataset = [
            {
                "input_ids": [0, 50, 51, 1, 10, 11], # BOS, C, C, SEP, P, P
                "raw_plaintext": "ab"
            }
        ]

        # Mock solve to return what we expect
        solver.solve = MagicMock(return_value="ab")

        solver.evaluate(mock_dataset)

        # Check if the jsonl file was created
        log_file = tmp_path / "evaluation_results.jsonl"
        assert log_file.exists()

        # Verify the content of the log
        with open(log_file) as f:
            data = json.loads(f.readline())
            assert data["target"] == "ab"
            assert data["pred"] == "ab"
            assert data["ser"] == 0.0

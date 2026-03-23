import pytest
import json
import torch
from unittest.mock import patch
import src.eval as eval_module

@pytest.fixture
def mock_eval_env(tmp_path):
    """Sets up a fake environment for evaluation testing."""
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()

    model_dir = tmp_path / "exp_latest"
    model_dir.mkdir()
    model_path = model_dir / "best.pth"
    model_path.write_text("fake_weights")

    return test_dir, model_path

@patch("src.eval.CipherSolver")
@patch("src.eval.DataLoader")
@patch("src.eval.CipherDataset")
@patch("src.eval.DataManager.scan_directory")
def test_test_model_logic(
    mock_scan,
    mock_dataset,
    mock_loader,
    mock_solver_class,
    mock_eval_env
):
    test_dir, model_path = mock_eval_env

    mock_solver = mock_solver_class.return_value
    mock_solver.decrypt.return_value = "hello"
    mock_solver.calculate_ser.return_value = 0.0

    # 1. Define your separator ID (must match what solver.tokenizer expects)
    # 1. Define a specific sep_id
    # 1. Setup the Solver Mock properly
    mock_solver = mock_solver_class.return_value

    # Ensure the tokenizer's decode method returns a real string
    mock_solver.tokenizer.decode.return_value = "hello"

    # Ensure the sep_token_id is a real integer
    test_sep_id = 100
    mock_solver.tokenizer.sep_token_id = test_sep_id

    # Ensure metrics are real numbers
    mock_solver.calculate_ser.return_value = 0.0
    mock_solver.decrypt.return_value = "hello_decrypted"

    # 2. Match your mock input to that SEP ID
    mock_input = torch.tensor([[1, 2, test_sep_id, 4, 5]])

    # 3. Use a real list of IDs/Strings in the batch
    fake_batch = {
        "input_ids": mock_input,
        "labels": torch.tensor([[-100, -100, -100, 10, 11]]),
        "id": ["sample_001"], # Real string, not a mock
    }
    mock_loader.return_value = [fake_batch]

    with patch("src.eval.config", create=True) as mock_config:
        mock_config.max_len = 10
        mock_config.save_path = str(model_path.parent.parent)

        eval_module.test_model(test_dir, model_path=model_path)

    expected_output = model_path.parent / f"eval_{model_path.stem}.jsonl"
    assert expected_output.exists()

    with open(expected_output) as f:
        lines = f.readlines()
        assert len(lines) > 0
        data = [json.loads(line) for line in lines]

    assert data[0]["predicted"] == "hello_decrypted"
    assert data[0]["ser"] == 0.0

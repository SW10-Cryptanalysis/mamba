import pytest
import json
import torch
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


def test_test_model_logic(mocker, mock_eval_env):
    test_dir, model_path = mock_eval_env

    mocker.patch("src.eval.Path.exists", return_value=True)
    mocker.patch("src.eval.PretokenizedCipherDataset")

    mock_loader = mocker.patch("src.eval.DataLoader")
    mock_solver_class = mocker.patch("src.eval.CipherSolver")

    mock_solver = mock_solver_class.return_value
    mock_solver.tokenizer.decode.return_value = "hello"

    test_sep_id = 100
    mock_solver.tokenizer.sep_token_id = test_sep_id

    mock_solver.calculate_ser.return_value = 0.0
    mock_solver.decrypt.return_value = "hello_decrypted"

    mock_input = torch.tensor([[1, 2, test_sep_id, 4, 5]])

    fake_batch = {
        "input_ids": mock_input,
        "labels": mock_input,
        "id": ["sample_001"],
    }
    mock_loader.return_value = [fake_batch]

    mock_config = mocker.patch("src.eval.config", create=True)
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

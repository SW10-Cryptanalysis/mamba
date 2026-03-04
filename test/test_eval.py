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
    
    fake_batch = (
        torch.zeros((1, 5)), 
        ["hello"],           
        {"path": ["/path/to/test.txt"], "internal_name": [None]}
    )
    mock_loader.return_value = [fake_batch]
    
    with patch("src.eval.config", create=True) as mock_config:
        mock_config.max_len = 10
        mock_config.save_path = str(model_path.parent.parent)
        
        eval_module.test_model(test_dir, model_path=model_path)
    
    expected_output = model_path.parent / f"eval_{model_path.stem}.json"
    assert expected_output.exists()
    
    with open(expected_output, "r") as f:
        data = json.load(f)
        assert data["predictions"][0]["predicted"] == "hello"
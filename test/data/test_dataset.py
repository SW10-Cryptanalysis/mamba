import pytest
import torch
from unittest.mock import patch
from src.data.dataset import CipherDataset
from src.data.tokenizer import CipherTokenizer
from src.config import Config

@pytest.fixture
def tokenizer():
    """Instantiate a real CipherTokenizer with a test configuration."""
    config = Config()
    config.unique_homophones = 1
    return CipherTokenizer(config)

@pytest.fixture
def sample_file_paths():
    return [("fake/path/1.json", None), ("fake/archive.zip", "sample_1")]

def test_dataset_length(sample_file_paths, tokenizer):
    """Verify that the dataset reports the correct number of samples."""
    dataset = CipherDataset(sample_file_paths, max_seq_len=10, tokenizer=tokenizer)
    assert len(dataset) == 2

@patch("src.utils.data_manager.DataManager.load_sample")
def test_dataset_train_mode(mock_load, sample_file_paths, tokenizer):
    """Verify that train mode returns (cipher_tensor, plain_tensor)."""
    mock_load.return_value = {
        "ciphertext": "1 2 3",
        "plaintext": "abc"
    }

    max_len = 5
    dataset = CipherDataset(sample_file_paths, max_seq_len=max_len, tokenizer=tokenizer, mode="train")
    cipher, plain = dataset[0]

    assert torch.is_tensor(cipher)
    assert cipher.shape[0] == max_len
    assert cipher.tolist() == [1, 2, 3, 0, 0]

    assert torch.is_tensor(plain)
    assert plain.shape[0] == max_len

    expected_ids = tokenizer.encode("abc")
    expected_padded = (expected_ids + [0] * max_len)[:max_len]

    assert plain.tolist() == expected_padded

@patch("src.utils.data_manager.DataManager.load_sample")
def test_dataset_eval_mode_robust(mock_load, sample_file_paths, tokenizer):
    """Verify eval mode returns correct types, handles None, and pads correctly."""
    mock_load.return_value = {
        "ciphertext": [10, 20],
        "plaintext": "hello"
    }

    max_len = 4
    dataset = CipherDataset(sample_file_paths, max_seq_len=max_len, tokenizer=tokenizer, mode="eval")
    cipher_tensor, plain_str, metadata = dataset[0]

    assert cipher_tensor.shape[0] == max_len
    assert cipher_tensor.tolist() == [10, 20, 0, 0]

    assert plain_str == "hello"

    assert metadata["internal_name"] == ""
    assert isinstance(metadata["path"], str)
    assert metadata["path"] == sample_file_paths[0][0]

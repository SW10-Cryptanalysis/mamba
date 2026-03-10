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
    """Verify train mode returns unpadded dict (collate_fn handles padding later)."""
    mock_load.return_value = {
        "ciphertext": "1 2 3",
        "plaintext": "abc"
    }

    dataset = CipherDataset(sample_file_paths, max_seq_len=10, tokenizer=tokenizer, mode="train")

    batch = dataset[0]
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    expected_content_len = 7

    assert input_ids.shape[0] == expected_content_len
    assert labels.shape[0] == expected_content_len

    assert input_ids.tolist() == [1, 2, 3, tokenizer.sep_token_id] + tokenizer.encode("abc")
    assert labels.tolist() == [-100, -100, -100, -100] + tokenizer.encode("abc")

@patch("src.utils.data_manager.DataManager.load_sample")
def test_dataset_eval_mode_robust(mock_load, sample_file_paths, tokenizer):
    """Verify eval mode returns Cipher + SEP without internal padding."""
    mock_load.return_value = {
        "ciphertext": [10, 20],
        "plaintext": "hello"
    }

    max_len = 10
    dataset = CipherDataset(sample_file_paths, max_seq_len=max_len, tokenizer=tokenizer, mode="eval")
    cipher_tensor, plain_str, metadata = dataset[0]

    assert torch.is_tensor(cipher_tensor)
    assert isinstance(plain_str, str)
    assert isinstance(metadata, dict)

    assert cipher_tensor.shape[0] == 3
    assert cipher_tensor[-1].item() == tokenizer.sep_token_id
    assert plain_str == "hello"

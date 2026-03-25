import pytest
import torch
from unittest.mock import patch
from src.data.dataset import PretokenizedCipherDataset
from src.config import Config
from pathlib import Path

@pytest.fixture
def config():
    """Test configuration."""
    conf = Config()
    conf.unique_homophones = 100
    conf.max_len = 10
    return conf

@pytest.fixture
def mock_arrow_data():
    """Creates a list of dicts simulating what Arrow returns."""
    return [
        {"input_ids": [1, 2, 3, 101, 106, 107], "labels": [1, 2, 3, 101, 106, 107]},
        {"input_ids": [10, 20, 101, 110], "labels": [10, 20, 101, 110]},
    ]

@patch("src.data.dataset.load_from_disk")
def test_dataset_length(mock_load, config, mock_arrow_data):
    """Verify that the dataset reports the correct number of samples from Arrow."""
    mock_load.return_value = mock_arrow_data
    dataset = PretokenizedCipherDataset(Path("fake/path"), max_seq_len=10, config=config)

    assert len(dataset) == 2

@patch("src.data.dataset.load_from_disk")
def test_dataset_getitem_content(mock_load, config, mock_arrow_data):
    """Verify __getitem__ returns variable-length tensors as expected."""
    mock_load.return_value = mock_arrow_data
    dataset = PretokenizedCipherDataset(Path("fake/path"), max_seq_len=10, config=config)

    batch = dataset[0]
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    assert isinstance(input_ids, torch.Tensor)
    assert input_ids.shape[0] == 6
    assert input_ids.tolist() == [1, 2, 3, 101, 106, 107]
    assert labels.tolist() == [1, 2, 3, 101, 106, 107]

@patch("src.data.dataset.load_from_disk")
def test_dataset_truncation(mock_load, config):
    """Verify that sequences longer than max_seq_len are truncated."""
    long_data = [{"input_ids": list(range(20)), "labels": list(range(20))}]
    mock_load.return_value = long_data
    dataset = PretokenizedCipherDataset(Path("fake/path"), max_seq_len=10, config=config)

    batch = dataset[0]
    assert batch["input_ids"].shape[0] == 10
    assert batch["labels"].shape[0] == 10

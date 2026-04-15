import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from src.data.dataset import CipherPlainData

@pytest.fixture
def mock_config(tmp_path: Path):
    """Creates a mock config with a unique, secure temporary directory."""
    config = MagicMock()

    mock_dir = tmp_path / "mock_data"
    mock_dir.mkdir()

    config.tokenized_dir = mock_dir
    return config

@pytest.fixture
def mock_dataset_content():
    """Mock content that simulate what load_from_disk returns."""
    return [
        {"input_ids": [1, 10, 11, 2, 20, 21], "labels": [1, 10, 11, 2, 20, 21]},
        {"input_ids": [1, 50, 2, 60], "labels": [1, 50, 2, 60]},
    ]

class TestCipherPlainData:

    @patch("src.data.dataset.load_from_disk")
    def test_init_success(self, mock_load, mock_config):
        """Test if dataset initializes correctly when path exists."""
        expected_path = mock_config.tokenized_dir / "Training"

        with patch.object(Path, "exists", return_value=True):
            mock_load.return_value = range(10)

            ds = CipherPlainData(mock_config.tokenized_dir, split="Training")

            mock_load.assert_called_once_with(str(expected_path))
            assert len(ds) == 10

    def test_init_file_not_found(self, mock_config):
        """Test if it raises FileNotFoundError when path is missing."""
        path = mock_config.tokenized_dir / "Training"
        with patch.object(Path, "exists", return_value=False), \
             pytest.raises(FileNotFoundError, match="run preprocess.py first"):

            CipherPlainData(path, split="Validation")

    @patch("src.data.dataset.load_from_disk")
    def test_getitem(self, mock_load, mock_config, mock_dataset_content):
        """Test if __getitem__ returns the correct TypedDict structure."""
        with patch.object(Path, "exists", return_value=True):
            mock_load.return_value = mock_dataset_content

            ds = CipherPlainData(mock_config)
            item = ds[0]

            assert isinstance(item, dict)
            assert "input_ids" in item
            assert "labels" in item

            assert item["input_ids"] == [1, 10, 11, 2, 20, 21]
            assert item["labels"] == [1, 10, 11, 2, 20, 21]

    @patch("src.data.dataset.load_from_disk")
    def test_len(self, mock_load, mock_config, mock_dataset_content):
        """Test the __len__ method."""
        with patch.object(Path, "exists", return_value=True):
            mock_load.return_value = mock_dataset_content
            ds = CipherPlainData(mock_config)
            assert len(ds) == 2

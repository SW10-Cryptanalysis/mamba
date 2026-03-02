import pytest
from unittest.mock import patch, mock_open
from src.data.dataset import CipherDataset
from src.utils.data_manager import DatasetManager
from src.config import Config

@pytest.fixture
def mock_config():
    """Provides a basic config for testing."""
    c = Config()
    c.unique_homophones = 100
    c.max_len = 10
    return c

def test_cipher_dataset_padding(mock_config):
    """Verify that the dataset correctly handles config and padding."""
    fake_files = [("fake_path.json", None)]
    
    with patch("src.utils.data_manager.DatasetManager.load_sample") as mock_load:
        mock_load.return_value = {
            "ciphertext": "1 2 3",
            "plaintext": "abc"
        }
        
        dataset = CipherDataset(fake_files, max_seq_len=5, config=mock_config)
        
        cipher, plain = dataset[0]
        
        assert cipher.shape[0] == 5
        assert cipher[0] == 1
        assert cipher[3] == 0
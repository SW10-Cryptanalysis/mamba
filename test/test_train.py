import os
import json
from unittest.mock import mock_open, patch
from src.train import process_json, CipherDataset

def test_process_json_list_input():
    """Verify it handles 'ciphertext' as a list of integers."""
    mock_data = {
        "ciphertext": [1, 5, 10, 2],
        "length": 4
    }
    mock_json = json.dumps(mock_data)
    
    with patch("builtins.open", mock_open(read_data=mock_json)):
        length, max_val = process_json("fake_path.json")
        
    assert length == 4
    assert max_val == 10

def test_cipher_dataset_padding():
    """Verify that the dataset correctly pads and truncates tensors."""
    temp_dir = "temp_test_data"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, "sample.json")
    
    with open(file_path, "w") as f:
        json.dump({
            "ciphertext": "1 2 3",
            "plaintext": "abc"
        }, f)

    dataset = CipherDataset(temp_dir, max_seq_len=5)
    cipher, plain = dataset[0]
    
    assert cipher.shape[0] == 5
    assert cipher[3] == 0
    offset = dataset.char_offset
    expected_plain = [0 + offset, 1 + offset, 2 + offset, 0, 0]
    assert plain.tolist() == expected_plain

    os.remove(file_path)
    os.rmdir(temp_dir)
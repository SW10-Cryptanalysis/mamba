import os
import json
from src.train import process_json, CipherDataset, get_max_stats

def test_process_json(tmp_path):
    """Verify that process_json correctly extracts stats from a dummy file."""
    d = tmp_path / "test.json"
    d.write_text(json.dumps({"length": 100, "num_symbols": 50}))
    
    length, symbols = process_json(str(d))
    assert length == 100
    assert symbols == 50

def test_cipher_dataset_padding():
    """Verify that the dataset correctly pads and truncates tensors."""
    temp_dir = "temp_test_data"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, "sample.json")
    
    with open(file_path, "w") as f:
        json.dump({
            "recurrence_encoding": "1 2 3",
            "plaintext": "abc"
        }, f)

    dataset = CipherDataset(temp_dir, max_seq_len=5)
    cipher, plain = dataset[0]
    
    assert cipher.shape[0] == 5
    assert cipher[3] == 0
    assert plain.tolist() == [0, 1, 2, 0, 0] 

    os.remove(file_path)
    os.rmdir(temp_dir)

def test_get_max_stats_cache(tmp_path):
    """Verify that get_max_stats creates a cache file."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "1.json").write_text(json.dumps({"length": 10, "num_symbols": 5}))
    
    max_len, max_sym = get_max_stats(str(data_dir))
    
    assert max_len == 10
    assert max_sym == 5
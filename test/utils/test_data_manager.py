import json
import zipfile
import pytest
from src.utils.data_manager import DataManager

@pytest.fixture
def data_dir(tmp_path):
    """Create a temporary directory with a mix of JSON and ZIP files."""
    # 1. Create a plain JSON file
    json_file = tmp_path / "sample1.json"
    with open(json_file, "w") as f:
        json.dump({"ciphertext": "1 2 30", "plaintext": "abc"}, f)

    # 2. Create a ZIP file containing a JSON
    zip_path = tmp_path / "archive.zip"
    inner_json_content = json.dumps({"ciphertext": [5, 10, 100], "plaintext": "xyz"})
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("inner_sample.json", inner_json_content)

    return tmp_path

def test_scan_directory(data_dir):
    """Ensure scanner finds both direct JSONs and those inside ZIPs."""
    results = DataManager.scan_directory(data_dir)

    # Should find sample1.json (None) and archive.zip (inner_sample.json)
    assert len(results) == 2

    # Check if internal_name logic is correct
    extensions = [r[1] for r in results]
    assert None in extensions
    assert "inner_sample.json" in extensions

def test_load_sample_plain(data_dir):
    """Test loading a standard JSON file."""
    path = str(data_dir / "sample1.json")
    data = DataManager.load_sample(path)
    assert data["plaintext"] == "abc"
    assert "1 2 30" in data["ciphertext"]

def test_load_sample_zip(data_dir):
    """Test loading a JSON file from inside a ZIP."""
    path = str(data_dir / "archive.zip")
    data = DataManager.load_sample(path, internal_name="inner_sample.json")
    assert data["plaintext"] == "xyz"
    assert data["ciphertext"] == [5, 10, 100]

def test_get_max_stats(data_dir):
    """Verify parallel stats calculation across different file types."""
    file_paths = DataManager.scan_directory(data_dir)

    # File 1: "1 2 30" -> len 3, max 30
    # File 2: [5, 10, 100] -> len 3, max 100
    max_len, max_sym = DataManager.get_max_stats(file_paths)

    assert max_len == 3
    assert max_sym == 100

def test_get_latest_checkpoint(tmp_path):
    """Verify finding the most recent checkpoint by mtime."""
    # Setup two experiment folders
    exp1 = tmp_path / "exp_1"
    exp2 = tmp_path / "exp_2"
    exp1.mkdir()
    exp2.mkdir()

    ckpt1 = exp1 / "latest.pth"
    ckpt2 = exp2 / "latest.pth"

    ckpt1.write_text("dummy1")
    # Ensure mtime is different
    import time
    time.sleep(0.1)
    ckpt2.write_text("dummy2")

    latest = DataManager.get_latest_checkpoint(tmp_path)
    assert latest == ckpt2

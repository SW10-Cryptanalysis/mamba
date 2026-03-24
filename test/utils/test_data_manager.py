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

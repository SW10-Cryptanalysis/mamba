import pytest
import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from src.engine.trainer import MambaTrainer

@pytest.fixture
def mock_config(tmp_path):
    """Provides a basic mock configuration."""
    config = MagicMock()
    config.save_path = tmp_path / "normal" / "run_1"
    config.outputs_dir = tmp_path / "normal"
    config.use_spaces = False
    config.epochs = 1
    config.batch_size = 4
    config.grad_accum = 1
    config.learning_rate = 1e-4
    config.save_step = 10
    config.pad_token_id = 0
    return config

@pytest.fixture
def trainer_with_mocks(mock_config):
    """Instantiates MambaTrainer with mocked external dependencies."""
    with patch("src.engine.trainer.get_model"), \
         patch("src.engine.trainer.CipherPlainData"), \
         patch("src.engine.trainer.PadCollator"), \
         patch("src.engine.trainer.Trainer"):

        trainer = MambaTrainer(mock_config, resume=False)
        return trainer

class TestMambaTrainer:

    def test_init_fresh_run(self, trainer_with_mocks, mock_config):
        """Verify that a fresh run initializes paths correctly."""
        assert not trainer_with_mocks.resume
        assert trainer_with_mocks.save_path == Path(mock_config.save_path)
        assert trainer_with_mocks.save_path.exists()

    @patch("src.engine.trainer.get_last_checkpoint")
    def test_resolve_resume_path_auto(self, mock_get_last, mock_config, tmp_path):
        """Test auto-detecting the latest run directory by forcing timestamps."""
        normal_dir = tmp_path / "normal"
        normal_dir.mkdir(parents=True)

        # Create 'old_run'
        old_run = normal_dir / "old_run"
        old_run.mkdir()

        # Create 'new_run'
        new_run = normal_dir / "new_run"
        new_run.mkdir()

        # FORCE timestamps: Set new_run to be 100 seconds newer than old_run
        now = time.time()
        os.utime(old_run, (now - 100, now - 100))
        os.utime(new_run, (now + 100, now + 100))

        mock_config.outputs_dir = tmp_path
        mock_config.use_spaces = False

        # Mocking everything to avoid the ImportError and setup issues
        with patch("src.engine.trainer.get_model"), \
             patch("src.engine.trainer.CipherPlainData"), \
             patch("src.engine.trainer.PadCollator"), \
             patch("src.engine.trainer.TrainingArguments"), \
             patch("src.engine.trainer.Trainer"):

            trainer = MambaTrainer(mock_config, resume=True)

            # Now this should pass because new_run is definitively the 'latest'
            assert "new_run" in str(trainer.save_path)

    def test_save_config(self, trainer_with_mocks, tmp_path):
        """Ensure _save_config writes the expected keys to JSON."""
        test_dir = tmp_path / "test_save"
        test_dir.mkdir()

        trainer_with_mocks.cfg.test_param = "hello"
        trainer_with_mocks._save_config(test_dir)

        config_file = test_dir / "project_config.json"
        assert config_file.exists()

        with open(config_file) as f:
            data = json.load(f)
            assert data["test_param"] == "hello"

    def test_load_config_sync(self, trainer_with_mocks):
        """Test that _load_config correctly overwrites current config attributes."""
        mock_ckpt_path = Path("/fake/path")
        fake_json = {"learning_rate": 0.99, "epochs": 100}

        # Mocking the open() call and the exists() check
        with patch("pathlib.Path.exists", return_value=True), \
             patch("builtins.open", mock_open(read_data=json.dumps(fake_json))):

            synced_cfg = trainer_with_mocks._load_config(trainer_with_mocks.cfg, mock_ckpt_path)

            assert synced_cfg.learning_rate == 0.99
            assert synced_cfg.epochs == 100

    def test_run_logic(self, trainer_with_mocks):
        """Verify the sequence of events in the run() method."""
        trainer_with_mocks.trainer.train = MagicMock()

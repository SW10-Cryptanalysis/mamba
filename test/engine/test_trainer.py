import json

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.engine.trainer import MambaTrainer
from src.config import Config

class SimpleLinearModel(nn.Module):
    """A tiny CPU-friendly model that mimics the MambaModel interface."""
    def __init__(self, vocab_size, char_offset):
        super().__init__()
        self.char_offset = char_offset
        self.embedding = nn.Embedding(vocab_size, 16)
        self.out = nn.Linear(16, vocab_size)

    def forward(self, x):
        return self.out(self.embedding(x))

@pytest.fixture
def cpu_trainer_setup(tmp_path):
    config = Config()
    config.learning_rate = 0.001
    vocab_size = 200
    char_offset = 100

    cipher = torch.randint(0, 100, (4, 5))
    plain = torch.randint(100, 200, (4, 5))
    loader = DataLoader(TensorDataset(cipher, plain), batch_size=2)

    save_path = tmp_path / "train_out"
    save_path.mkdir()

    model = SimpleLinearModel(vocab_size, char_offset)

    trainer = MambaTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        config=config,
        save_path=save_path,
        device="cpu"
    )
    return trainer

def test_trainer_step_logic(cpu_trainer_setup):
    """Verify the backprop and weight update logic works on CPU."""
    trainer = cpu_trainer_setup

    for param_group in trainer.optimizer.param_groups:
        param_group['lr'] = 1.0

    # Capture weights before a step
    params = list(trainer.model.parameters())
    initial_weight = params[0].clone().detach()

    # Run one epoch
    loss = trainer._train_one_epoch()

    # Verify weights changed
    updated_weight = params[0].detach()
    assert not torch.equal(initial_weight, updated_weight)
    assert loss > 0

def test_history_logging(cpu_trainer_setup):
    """Verify that history dict is populated correctly."""
    config = Config
    trainer = cpu_trainer_setup
    trainer.train(epochs=2)

    assert len(trainer.history["train_loss"]) == 2
    assert len(trainer.history["learning_rates"]) == 2
    assert trainer.history["learning_rates"][0] == config.learning_rate

def test_save_config(cpu_trainer_setup):
    """Verify that configuration is correctly serialized to JSON."""
    trainer = cpu_trainer_setup
    config_path = trainer.exp_dir / "config.json"

    assert config_path.exists()

    with open(config_path) as f:
        saved_data = json.load(f)

    assert "learning_rate" in saved_data
    assert saved_data["learning_rate"] == trainer.config.learning_rate

def test_load_checkpoint_and_resume(cpu_trainer_setup, tmp_path):
    """Verify trainer can restore state from a .pth file and history.json."""
    trainer = cpu_trainer_setup

    checkpoint_path = trainer.exp_dir / "manual_check.pth"
    state = {
        "epoch": 5,
        "model_state_dict": trainer.model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "scheduler_state_dict": trainer.scheduler.state_dict(),
        "val_loss": 0.1234
    }
    torch.save(state, checkpoint_path)

    fake_history = {"train_loss": [0.5, 0.4], "val_loss": [0.5, 0.4], "learning_rates": [0.01, 0.01]}
    with open(trainer.exp_dir / "history.json", "w") as f:
        json.dump(fake_history, f)

    trainer.load_checkpoint(checkpoint_path)

    assert trainer.current_epoch == 5
    assert trainer.best_val_loss == 0.1234
    assert len(trainer.history["train_loss"]) == 2
    assert trainer.history["train_loss"][0] == 0.5

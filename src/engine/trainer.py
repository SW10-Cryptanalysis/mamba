from dataclasses import asdict
import json
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from src.config import Config
from src.utils.logging import get_logger
logger = get_logger("engine/trainer.py")

class MambaTrainer:
    """Trainer class for Mamba-based cipher models.

    This class encapsulates the training loop, validation logic, checkpointing,
    and learning rate scheduling. It automatically handles experiment directory
    creation and configuration logging.

    Attributes:
        model (nn.Module): The Mamba model to be trained.
        train_loader (DataLoader): Iterable for the training dataset.
        val_loader (DataLoader): Iterable for the validation dataset.
        config (Config): Configuration object containing hyperparameters.
        save_path (Path): Base directory where all experiments are saved.
        exp_dir (Path): Specific directory for the current experiment run.
        device (str): Computation device (e.g., 'cuda' or 'cpu').
        criterion (nn.Module): The loss function (CrossEntropyLoss).
        optimizer (optim.Optimizer): The AdamW optimizer.
        scheduler (optim.lr_scheduler): Learning rate scheduler.
        history (dict): Log of losses and learning rates throughout training.

    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config,
        save_path: Path,
        exp_dir: Path = None,
        device: str = "cuda",
    ) -> None:
        """Initialize the trainer with model, data, and experimental settings.

        Args:
            model: The neural network model to train.
            train_loader: DataLoader providing training samples.
            val_loader: DataLoader providing validation samples.
            config: Config instance containing training hyperparameters.
            save_path: Path where experiment folders will be created.
            exp_dir: Optional path to an existing experiment directory
                (used for resuming).
            device: Device to use for training. Defaults to "cuda".

        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.save_path = save_path
        self.device = device

        if exp_dir:
            self.exp_dir = exp_dir
            self.timestamp = exp_dir.name.replace("exp_", "")
        else:
            self.timestamp = datetime.now().strftime("%d%m_%H%M_%Y")
            self.exp_dir = save_path / f"exp_{self.timestamp}"
            self.exp_dir.mkdir(parents=True, exist_ok=True)
            self._save_config()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.factor,
            patience=config.patience,
        )

        self.history = {"train_loss": [], "val_loss": [], "learning_rates": []}

        self.best_val_loss = float("inf")
        self.current_epoch = 0

    def _save_config(self) -> None:
        """Save the current configuration to a JSON file."""
        config_path = self.exp_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(asdict(self.config), f, indent=4, default=str)
        logger.info(f"Experiment config saved to {config_path}")

    def _save_history(self) -> None:
        """Save the current training history to a JSON file."""
        history_path = self.exp_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=4)
        logger.info(f"History updated at {history_path}")

    def _save_checkpoint(self, val_loss: float, is_best: bool) -> None:
        """Save a model checkpoint and update the 'latest' and 'best' files.

        Args:
            val_loss: The validation loss achieved in the current epoch.
            is_best: A flag indicating if this model achieved the lowest
                validation loss seen so far.

        """
        state = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "char_offset": self.model.char_offset,
            "d_model": self.config.d_model,
            "n_layers": self.config.n_layers,
        }

        epoch_filename = f"epoch_{self.current_epoch:03d}.pth"
        epoch_path = self.exp_dir / epoch_filename
        torch.save(state, epoch_path)

        latest_path = self.exp_dir / "latest.pth"
        shutil.copyfile(epoch_path, latest_path)

        if is_best:
            best_path = self.exp_dir / "best.pth"
            shutil.copyfile(epoch_path, best_path)
            logger.info(
                f"Updated best model at epoch {self.current_epoch} "
                f"(Loss: {val_loss:.4f})",
            )

    def load_checkpoint(self, checkpoint_path: Path) -> "MambaTrainer":
        """Restore the trainer state from a saved checkpoint file.

        Args:
            checkpoint_path: The filesystem path to the .pth checkpoint file.

        Returns:
            The MambaTrainer instance (self).

        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint.get("val_loss", float("inf"))

        if "scheduler_state_dict" in checkpoint:
             self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        history_path = self.exp_dir / "history.json"
        if history_path.exists():
            with open(history_path) as f:
                self.history = json.load(f)

        return self

    def train(self, epochs: int) -> None:
        """Execute the main training loop for a specified number of epochs.

        Args:
            epochs: The total number of epochs to train for.

        """
        start_epoch = self.current_epoch

        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch + 1

            avg_train_loss = self._train_one_epoch()
            avg_val_loss = self._validate_one_epoch()

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(avg_val_loss)

            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(avg_val_loss)
            self.history["learning_rates"].append(current_lr)

            self._save_history()

            logger.info(
                f"Epoch [{self.current_epoch}/{epochs}] - "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}",
            )

            is_best = avg_val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = avg_val_loss

            self._save_checkpoint(avg_val_loss, is_best)

            if current_lr < 1e-7:
                logger.info("Learning rate too low. Stopping early.")
                break

    def _train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        loop = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch} [Train]",
            leave=False,
        )

        for cipher, plain in loop:
            cipher, plain = cipher.to(self.device), plain.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(cipher)

            loss = self.criterion(outputs.view(-1, outputs.size(-1)), plain.view(-1))
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        return total_loss / len(self.train_loader)

    def _validate_one_epoch(self) -> float:
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for cipher, plain in self.val_loader:
                cipher, plain = cipher.to(self.device), plain.to(self.device)
                outputs = self.model(cipher)
                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)), plain.view(-1),
                )
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

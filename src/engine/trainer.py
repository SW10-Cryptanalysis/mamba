from dataclasses import asdict
import json
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import math
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
        run_type: str = "normal",
        device: str = "cuda",
    ) -> None:
        """Initialize the trainer with model, data, and experimental settings.

        Args:
            model: The neural network model to train.
            train_loader: DataLoader providing training samples.
            val_loader: DataLoader providing validation samples.
            config: Config instance containing training hyperparameters.
            run_type: Normal or spaced training,
            exp_dir: Optional path to an existing experiment directory
                (used for resuming).
            device: Device to use for training. Defaults to "cuda".

        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.run_type = run_type
        self.device = device
        self.resume_step = 0

        if exp_dir:
            self.exp_dir = exp_dir
            self.timestamp = exp_dir.name.replace("exp_", "")
        else:
            self.timestamp = datetime.now().strftime("%d%m_%H%M_%Y")
            self.exp_dir = config.save_path / f"exp_{self.run_type}_{self.timestamp}"
            self.exp_dir.mkdir(parents=True, exist_ok=True)
            self._save_config()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate)

        self.total_steps = len(self.train_loader) * self.config.epochs
        self.warmup_steps = int(self.total_steps * 0.1)
        self.decay_start_step = int(self.total_steps * 0.9)

        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._get_wsd_schedule,
        )

        self.history = {"train_loss": [], "val_loss": [], "learning_rates": []}

        self.best_val_loss = float("inf")
        self.current_epoch = 0

        total, trainable = self.count_parameters(self.model)
        logger.info(f"Total Parameters: {total:,}")
        logger.info(f"Trainable Parameters: {trainable:,}")

    def _get_wsd_schedule(self, current_step: int) -> float:
        """Calculate the LR multiplier for Warmup-Stable-Decay.

        Args:
            current_step: How many steps of training have passed.

        """
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1, self.warmup_steps))

        if current_step < self.decay_start_step:
            return 1.0

        progress = float(current_step - self.decay_start_step) / float(
            max(1, self.total_steps - self.decay_start_step),
        )

        return 0.5 * (1.0 + math.cos(math.pi * progress))

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

    def _save_checkpoint(
        self,
        val_loss: float,
        is_best: bool,
        suffix: str = None,
    ) -> None:
        """Save a model checkpoint.

        Args:
            val_loss: Current loss.
            is_best: If true, copies to best.pth.
            suffix: Optional string (e.g., 'step_5000') for intra-epoch saves.

        """
        state = {
            "epoch": self.current_epoch,
            "step": getattr(self, "current_step", 0),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "char_offset": self.model.char_offset,
            "d_model": self.config.d_model,
            "n_layers": self.config.n_layers,
        }

        base_name = f"epoch_{self.current_epoch:03d}"
        if suffix:
            base_name += f"_{suffix}"

        epoch_filename = f"{base_name}.pth"
        epoch_path = self.exp_dir / epoch_filename
        torch.save(state, epoch_path)

        latest_path = self.exp_dir / "latest.pth"
        shutil.copyfile(epoch_path, latest_path)

        if is_best and not suffix:
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
        self.resume_step = checkpoint.get("step", 0)
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
            self.current_epoch = epoch

            avg_train_loss = self._train_one_epoch()
            avg_val_loss = self._validate_one_epoch()

            current_lr = self.optimizer.param_groups[0]["lr"]

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

            self.current_epoch = epoch + 1
            self._save_checkpoint(avg_val_loss, is_best)

            if current_lr < 1e-7:
                logger.info("Learning rate too low. Stopping early.")
                break

    def _train_one_epoch(self) -> float:
        """Run a single epoch of training, processing batches from the training loader.

        Returns:
            float: The average training loss across all processed batches in
                the current epoch.

        """
        self.model.train()
        total_loss, batches_processed = 0, 0
        resume_step = getattr(self, "resume_step", 0)

        loop = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1} [Train]",
            leave=False,
        )

        for i, batch in enumerate(loop):
            if i < resume_step:
                continue

            loss_value = self._execute_training_step(batch, i)

            total_loss += loss_value
            batches_processed += 1

            self._update_training_ui(loop, i, loss_value)

            step = i + 1
            if step % self.config.save_step == 0:
                self._handle_intermediate_checkpoint(step, loss_value)

        self.resume_step = 0
        return total_loss / max(1, batches_processed)

    def _validate_one_epoch(self) -> float:
        """Evaluate the model on the validation dataset for one epoch.

        Returns:
            float: The average validation loss across all batches in the
                validation loader.

        """
        self.model.eval()
        total_loss = 0
        loop = tqdm(
            self.val_loader,
            desc=f"Epoch {self.current_epoch} [Val]",
            leave=False,
        )

        with torch.no_grad():
            for batch in loop:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = self._compute_batch_loss(batch)
                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())

        return total_loss / len(self.val_loader)

    def _compute_batch_loss(
        self,
        batch: dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the loss for a single batch.

        Args:
            batch: A batch of data, either as a dictionary containing "input_ids"
                and "labels", or a tuple/list in the form (input_ids, labels).

        Returns:
            torch.Tensor: A scalar tensor representing the CrossEntropy loss
                for the batch.

        """
        if not isinstance(batch, dict):
            input_ids, labels = batch
        else:
            input_ids = batch["input_ids"]
            labels = batch["labels"]

        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        logits = self.model(input_ids)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = self.criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        return loss

    def _execute_training_step(
        self,
        batch: dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor],
        step_idx: int,
    ) -> float:
        """Execute a single pass for a training batch.

        Args:
            batch: A batch of data containing "input_ids" and "labels".
            step_idx: The current iteration index within the current epoch.

        Returns:
            float: The scalar loss value for the current batch.

        """
        global_step = (self.current_epoch * len(self.train_loader)) + step_idx
        self.current_step = step_idx

        self.optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = self._compute_batch_loss(batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step(global_step)

        return loss.item()

    def _update_training_ui(self, loop: tqdm, step_idx: int, loss: float) -> None:
        """Calculate the current training phase and updates the progress bar.

        Args:
            loop: The active tqdm progress bar instance for the training epoch.
            step_idx: The current iteration index within the current epoch.
            loss: The scalar loss value from the most recent training step to display.

        """
        global_step = (self.current_epoch * len(self.train_loader)) + step_idx

        if global_step < self.warmup_steps:
            phase = "warmup"
        elif global_step < self.decay_start_step:
            phase = "stable"
        else:
            phase = "decay"

        current_lr = self.optimizer.param_groups[0]["lr"]

        loop.set_postfix({
            "loss": f"{loss:.3f}",
            "lr": f"{current_lr:.2e}",
            "phase": phase,
        })

    def _handle_intermediate_checkpoint(self, step: int, loss: float) -> None:
        """Manage the 'sliding window' of checkpoints to save disk space.

        Args:
            step: The current iteration count (1-indexed) within the epoch.
            loss: The training loss at the current step, used for checkpoint metadata.

        """
        prev_step = step - self.config.save_step
        if prev_step > 0:
            ckpt_name = f"epoch_{self.current_epoch:03d}_step_{prev_step}.pth"
            prev_checkpoint = self.exp_dir / ckpt_name
            if prev_checkpoint.exists():
                prev_checkpoint.unlink()
                logger.info(f"Cleanup: Removed {prev_checkpoint.name}")

        logger.info(f"Step {step}: Saving intermediate checkpoint...")
        self._save_checkpoint(
            val_loss=loss,
            is_best=False,
            suffix=f"step_{step}",
        )

    def count_parameters(self, model: nn.Module) -> tuple[int, int]:
        """Count parameters for model."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return total_params, trainable_params

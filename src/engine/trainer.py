import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import logging

logger = logging.getLogger("engine/trainer.py")

class MambaTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        save_path: Path,
        device: str = "cuda"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.save_path = save_path
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.factor,
            patience=config.patience
        )
        
        self.best_val_loss = float("inf")
        self.current_epoch = 0

    def train(self, epochs: int):
        """Main entry point for training."""
        for epoch in range(epochs):
            self.current_epoch = epoch + 1
            
            avg_train_loss = self._train_one_epoch()
            avg_val_loss = self._validate_one_epoch()
            
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(avg_val_loss)

            logger.info(
                f"Epoch [{self.current_epoch}/{epochs}] - "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}"
            )

            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.save_best_model(avg_val_loss)

            if current_lr < 1e-7:
                logger.info("Learning rate too low. Stopping early.")
                break

    def _train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        loop = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [Train]", leave=False)

        for cipher, plain in loop:
            cipher, plain = cipher.to(self.device), plain.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(cipher)
            
            # Flatten outputs and targets for CrossEntropy
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
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), plain.view(-1))
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def save_best_model(self, val_loss: float):
        """Encapsulated saving logic with metadata."""
        state = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "char_offset": self.model.char_offset,
            "d_model": self.config.d_model,
            "n_layers": self.config.n_layers,
        }
        torch.save(state, self.save_path)
        logger.info(f"Saved new best model to {self.save_path.name}")
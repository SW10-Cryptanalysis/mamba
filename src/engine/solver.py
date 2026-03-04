from typing import Any

import torch
from pathlib import Path
from src.models.mamba import MambaModel
from src.data.tokenizer import CipherTokenizer
from src.config import Config
from src.utils.logging import get_logger
logger = get_logger("engine/solver.py")

class CipherSolver:
    """Interface for the Mamba Cipher model solver.

    Encapsulates model loading, inference, and decoding.

    Attributes:
        config (Config): Configuration object containing model and tokenizer settings.
        device (str): The device (cuda/cpu) where the model is loaded.
        tokenizer (CipherTokenizer): Tokenizer instance for handling character mapping.
        model (MambaModel | None): The underlying Mamba neural network,
            initialized during load_checkpoint.
        metadata (dict[str, Any]): Stores information about the loaded checkpoint,
            such as file path and validation loss.

    """

    def __init__(self, config: Config, device: str = "cuda") -> None:
        """Initialize the solver with configuration and device.

        Args:
            config: Configuration object for model and tokenizer parameters.
            device: Hardware device to run inference on (e.g., 'cuda' or 'cpu').

        """
        self.config = config
        self.device = device
        self.tokenizer = CipherTokenizer(config)
        self.model: MambaModel | None = None
        self.metadata: dict[str, Any] = {}

    def load_checkpoint(self, checkpoint_path: Path) -> "CipherSolver":
        """Model instantiation and weight loading from a .pth file.

        Args:
            checkpoint_path: Path to the PyTorch checkpoint file.

        Returns:
            The instance of CipherSolver for method chaining.

        Raises:
            FileNotFoundError: If the checkpoint path does not exist.

        """
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device,
                                weights_only=False)

        char_offset = checkpoint["char_offset"]
        vocab_size = char_offset + self.config.plain_vocab_size + self.config.buffer

        logger.debug(
            "Instantiating MambaModel: "
            f"vocab_size={vocab_size}, char_offset={char_offset}",
        )

        self.model = MambaModel(
            vocab_size=vocab_size,
            char_offset=char_offset,
            config=self.config,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.metadata = {"path": str(checkpoint_path),
                         "val_loss": checkpoint.get("val_loss")}
        return self

    @torch.no_grad()
    def decrypt(self, ciphertext: list[int] | str) -> str:
        """High-level API: Take raw ciphertext and return a human-readable string.

        Args:
            ciphertext: A list of integers or a space-separated string of homophones.

        Returns:
            The decrypted plaintext string truncated to match input length.

        Raises:
            RuntimeError: If called before load_checkpoint().

        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")

        if isinstance(ciphertext, str):
            ciphertext = [int(x) for x in ciphertext.split()]

        logger.debug(f"Decrypting sequence of length {len(ciphertext)}")

        input_tensor = (
            self.tokenizer.pad_sequence(ciphertext, self.config.max_len)
            .unsqueeze(0)
            .to(self.device)
        )

        logits = self.model(input_tensor)
        pred_indices = torch.argmax(logits, dim=-1).squeeze(0).tolist()

        full_decoded = self.tokenizer.decode(pred_indices)

        return full_decoded[:len(ciphertext)]

    def calculate_ser(self, pred: str, target: str) -> float:
        """Calculate Symbol Error Rate (SER).

        Args:
            pred: The predicted plaintext string.
            target: The ground truth plaintext string.

        Returns:
            The error rate as a float (0.0 is perfect, 1.0 is total error).

        """
        count = sum(1 for p, t in zip(pred, target, strict=False) if p == t)
        return 1.0 - (count / len(target))

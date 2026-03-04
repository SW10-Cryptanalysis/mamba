import torch
from pathlib import Path
from src.models.mamba import MambaModel
from src.data.tokenizer import CipherTokenizer
from src.utils.logging import get_logger
logger = get_logger("engine/solver.py")

class CipherSolver:
    """A production-ready interface for the Mamba Cipher model.
    Encapsulates model loading, inference, and decoding.
    """

    def __init__(self, config, device: str = "cuda"):
        self.config = config
        self.device = device
        self.tokenizer = CipherTokenizer(config)
        self.model = None
        self.metadata = {}

    def load_checkpoint(self, checkpoint_path: Path):
        """Automates model instantiation and weight loading from a .pth file."""
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        char_offset = checkpoint["char_offset"]
        vocab_size = char_offset + self.config.plain_vocab_size + self.config.buffer

        logger.debug(f"Instantiating MambaModel: vocab_size={vocab_size}, char_offset={char_offset}")

        self.model = MambaModel(
            vocab_size=vocab_size,
            char_offset=char_offset,
            config=self.config,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.metadata = {"path": str(checkpoint_path), "val_loss": checkpoint.get("val_loss")}
        return self

    @torch.no_grad()
    def decrypt(self, ciphertext: list[int] | str) -> str:
        """High-level API: Takes raw ciphertext and returns a human-readable string.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")

        if isinstance(ciphertext, str):
            ciphertext = [int(x) for x in ciphertext.split()]

        logger.debug(f"Decrypting sequence of length {len(ciphertext)}")

        input_tensor = self.tokenizer.pad_sequence(ciphertext, self.config.max_len).unsqueeze(0).to(self.device)

        logits = self.model(input_tensor)
        pred_indices = torch.argmax(logits, dim=-1).squeeze(0).tolist()

        full_decoded = self.tokenizer.decode(pred_indices)

        return full_decoded[:len(ciphertext)]

    def calculate_ser(self, pred: str, target: str) -> float:
        """Calculate Symbol Error Rate (SER)."""
        count = sum(1 for p, t in zip(pred, target) if p == t)
        return 1.0 - (count / len(target))

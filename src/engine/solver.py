from typing import Any
import torch
from pathlib import Path
from src.models.mamba import MambaModel
from mamba_ssm.utils.generation import InferenceParams
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
    def decrypt(self, input_ids: list[int] | torch.Tensor) -> str:
        """Performs autoregressive decryption of ciphertext using the Mamba model.

        This method handles both legacy ciphertext-only inputs and unified sequences 
        containing a separator.

        Args:
            input_ids: The sequence to decrypt. Can be a list of integer token IDs 
                or a torch.Tensor. If a unified sequence (Cipher + SEP 
                + Plain) is provided, tokens after the SEP are discarded before 
                generation begins.

        Returns:
            str: The decrypted plaintext string, decoded via the tokenizer.

        Note:
            The method automatically appends a SEP token if one is not present in 
            the input, signaling the model to begin the transition from cipher 
            processing to plaintext generation.

        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")

        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        elif input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0).to(self.device)

        sep_id = self.tokenizer.sep_token_id
        if sep_id in input_ids[0]:
            sep_idx = (input_ids[0] == sep_id).nonzero(as_tuple=True)[0][0]
            input_ids = input_ids[:, :sep_idx + 1]
        else:
            sep_tensor = torch.tensor([[sep_id]], device=self.device)
            input_ids = torch.cat([input_ids, sep_tensor], dim=1)

        cipher_len = input_ids.size(1) - 2
        inference_params = InferenceParams(max_seqlen=self.config.max_len, max_batch_size=1)

        logits = self.model(input_ids, inference_params=inference_params)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).view(1, 1).to(self.device)

        generated_tokens = []
        if next_token.item() != self.tokenizer.eos_token_id:
            generated_tokens.append(next_token.item())

        for _ in range(cipher_len - 1):
            logits = self.model(next_token, inference_params=inference_params)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).view(1, 1).to(self.device)

            token_id = next_token.item()
            if token_id == self.tokenizer.eos_token_id:
                break
            generated_tokens.append(token_id)

        return self.tokenizer.decode(generated_tokens)

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

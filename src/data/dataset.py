from typing import Literal
import torch
from torch.utils.data import Dataset
from ..utils.data_manager import DataManager
from src.data.tokenizer import CipherTokenizer
from src.utils.logging import get_logger
logger = get_logger("data/dataset.py")

class CipherDataset(Dataset):
    """A unified Dataset for Cipher tasks.

    Attributes:
        file_paths (list[tuple[str, str | None]]): List of tuples containing the
            filesystem path and an optional internal identifier for each sample.
        max_seq_len (int): The fixed length to which all sequences are padded
            or truncated.
        tokenizer (CipherTokenizer): The tokenizer used to transform strings
            into tensor indices.
        mode (Literal["train", "eval"]): Determines the return structure of __getitem__.

    """

    def __init__(
        self,
        file_paths: list[tuple[str, str | None]],
        max_seq_len: int,
        tokenizer: CipherTokenizer,
        mode: Literal["train", "eval"] = "train",
    ) -> None:
        """Initialize the dataset with file paths and configuration.

        Args:
            file_paths: List of (path, internal_name) tuples.
            max_seq_len: Maximum sequence length for padding/truncation.
            tokenizer: The tokenizer for encoding/decoding.
            mode: The operational mode. Must be one of 'train' or 'eval'.
                Defaults to 'train'.

        """
        self.file_paths = file_paths
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.mode = mode

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.file_paths)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, str, dict]:
        """Fetch a single sample from the dataset by index.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            In 'train' mode:
                tuple: (cipher_tensor, plain_tensor)
            In 'eval' mode:
                tuple: (cipher_tensor, original_plaintext, metadata)

        Raises:
            Exception: If there is an error loading the sample via DataManager.

        """
        path, internal_name = self.file_paths[idx]

        try:
            data = DataManager.load_sample(path, internal_name)

            ciphertext = data["ciphertext"]
            if isinstance(ciphertext, str):
                ciphertext = [int(x) for x in ciphertext.split()]

            cipher_tensor = self.tokenizer.pad_sequence(ciphertext, self.max_seq_len)

            if self.mode == "eval":
                metadata = {
                    "path": str(path),
                    "internal_name": internal_name if internal_name is not None else "",
                }
                return cipher_tensor, data["plaintext"], metadata

            encoded_plain = self.tokenizer.encode(data["plaintext"])
            plain_tensor = self.tokenizer.pad_sequence(encoded_plain, self.max_seq_len)

            return cipher_tensor, plain_tensor

        except Exception as e:
            logger.error(f"Error loading index {idx}: {e}")
            raise e

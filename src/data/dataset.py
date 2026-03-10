from typing import Literal
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk
from src.config import Config
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
    ) -> dict[str, torch.Tensor] | tuple[torch.Tensor, str, dict]:
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

            if self.mode == "eval":
                sep_id = self.tokenizer.sep_token_id
                eval_input = ciphertext + [sep_id]

                metadata = {
                    "path": str(path),
                    "internal_name": internal_name if internal_name is not None else "",
                }

                return torch.tensor(eval_input, dtype=torch.long), data["plaintext"], metadata

            encoded_plain = self.tokenizer.encode(data["plaintext"])
            sep_id = self.tokenizer.sep_token_id

            full_input = ciphertext + [sep_id] + encoded_plain

            full_labels = ([-100] * (len(ciphertext) + 1)) + encoded_plain

            full_input = full_input[:self.max_seq_len]
            full_labels = full_labels[:self.max_seq_len]

            return {
                "input_ids": torch.tensor(full_input, dtype=torch.long),
                "labels": torch.tensor(full_labels, dtype=torch.long),
            }

        except Exception as e:
            logger.error(f"Error loading index {idx}: {e}")
            raise e

class PretokenizedCipherDataset(Dataset):
    def __init__(self, directory_path, max_seq_len, config: Config):
        self.dataset = load_from_disk(str(directory_path))
        self.max_seq_len = max_seq_len
        self.sep_token_id = config.unique_homophones + 1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        input_ids = item["input_ids"]
        labels = item["labels"]

        sep_id = self.sep_token_id

        input_list = input_ids.tolist() if torch.is_tensor(input_ids) else list(input_ids)

        if sep_id in input_list:
            sep_idx = input_list.index(sep_id)
            new_labels = ([-100] * (sep_idx + 1)) + input_list[sep_idx + 1:]
        else:
            new_labels = labels

        return {
            "input_ids": torch.tensor(input_list, dtype=torch.long),
            "labels": torch.tensor(new_labels, dtype=torch.long),
        }

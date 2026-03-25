from pathlib import Path
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk
from src.config import Config
from src.utils.logging import get_logger
logger = get_logger("data/dataset.py")

class PretokenizedCipherDataset(Dataset):
    """A Dataset for handling pre-tokenized cipher sequences stored in Arrow format.

    This dataset is designed for models that use a unified input sequence where the
    ciphertext and plaintext are separated by a specific [SEP] token. It manages
    the masking of labels to ensure the model only calculates loss on the
    predicted plaintext tokens.

    Attributes:
        dataset: The underlying Hugging Face/Arrow dataset loaded from disk.
        max_seq_len (int): The fixed length to which all sequences are padded
            or truncated.
        sep_token_id (int): The unique token ID used to separate cipher
            and plain text.

    """

    def __init__(
        self,
        directory_path: Path,
        max_seq_len: int,
        config: Config,
    ) -> None:
        """Initialize the dataset with file paths and configuration.

        Args:
            directory_path: Path to the directory containing the pre-tokenized
                Arrow dataset files.
            max_seq_len: The fixed sequence length for the model. Inputs will
                be padded or truncated to this value.
            config: Configuration object used to derive the separator token ID
                based on the number of unique homophones.

        """
        self.dataset = load_from_disk(str(directory_path))
        self.max_seq_len = max_seq_len
        self.sep_token_id = config.unique_homophones + 1

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Retrieve a sample, truncate to max_seq_len, reutrn raw tensors.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            dict: {
                "input_ids": LongTensor of variable length <= max_seq_len,
                "labels": LongTensor of variable length <= max_seq_len.
            }

        """
        item = self.dataset[idx]
        input_ids = item["input_ids"]
        labels = item["labels"]

        input_list = (
            input_ids.tolist()
            if isinstance(input_ids, torch.Tensor)
            else list(input_ids)
        )

        label_list = (
            labels.tolist() if isinstance(labels, torch.Tensor) else list(labels)
        )

        input_list = input_list[:self.max_seq_len]
        label_list = label_list[:self.max_seq_len]

        return {
            "input_ids": torch.tensor(input_list, dtype=torch.long),
            "labels": torch.tensor(label_list, dtype=torch.long),
        }

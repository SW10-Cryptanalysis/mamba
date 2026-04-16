from typing import TypedDict
from torch.utils.data import Dataset
from pathlib import Path
from datasets import load_from_disk

class CipherPlainDataItem(TypedDict):
    """TypedDict for CipherPlainDataItem."""

    input_ids: list[int]
    labels: list[int]

class CipherPlainData(Dataset):
    """CipherPlainData dataset.

    This class is a subclass of torch.utils.data.Dataset and is used to load and
    iterate over the ciphertext-plaintext pairs in the Ciphers dataset.

    Attributes:
        dataset (Union[Dataset, DatasetDict]): loaded dataset.

    """

    def __init__(self, dataset_path: Path, split: str = "Training") -> None:
        """Initialize the CipherPlainData dataset.

        Args:
            dataset_path (Path): Path to the tokenized dataset.
            split (str): The data split to load (e.g., 'Training', 'Test').

        """
        path = dataset_path / split

        if not path.exists():
            raise FileNotFoundError(
                f"Missing Arrow Data: {path} - run preprocess.py first.",
            )

        self.dataset = load_from_disk(str(path))

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            int: The length of the dataset.

        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> CipherPlainDataItem:
        """Get the raw, unpadded item at the given index."""
        item = self.dataset[idx]

        return {
            "input_ids": item["input_ids"],
            "labels": item["labels"],
        }

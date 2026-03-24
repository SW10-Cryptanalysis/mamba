from pathlib import Path
import torch
from src.utils.logging import get_logger
logger = get_logger("utils/data_manager.py")

class DataManager:
	"""Utility class for dataset indexing, parsing, and statistical analysis.

	Provides a centralized interface for handling both raw JSON files and
	compressed ZIP archives, supporting multi-processed statistics calculation.

	Attributes:
		logger (Logger): The logging instance for tracking data operations.

	"""

	@staticmethod
	def get_latest_checkpoint(base_path: Path, prefix: str = "exp_*") -> Path | None:
		"""Search all exp_* folders for the newest latest.pth file.

		Args:
			base_path: The root directory containing experiment folders.
			prefix: prefix of which experiment was conducted.

		Returns:
			The path to the most recent checkpoint, or None if none are found.

		"""
		checkpoints = list(base_path.glob(f"{prefix}/latest.pth"))

		if not checkpoints:
			return None

		return max(checkpoints, key=lambda p: p.stat().st_mtime)


	@staticmethod
	def safe_pad_collate(
		batch: list[dict[str, torch.Tensor | list[int]]],
		pad_token_id: int = 0,
		ignore_index: int = -100,
	) -> dict[str, torch.Tensor]:
		"""Pad sequences to a multiple of 8.

		Args:
			batch: A list of dictionaries from the Dataset, where each dict contains
				"input_ids" and "labels".
			pad_token_id: The ID used to pad the `input_ids`. Defaults to 0.
			ignore_index: The value used to pad `labels`, signaling the loss
				function to ignore these positions. Defaults to -100.

		Returns:
			dict[str, torch.Tensor]: A dictionary containing:
				- "input_ids": Padded tensor of shape [batch_size, aligned_seq_len].
				- "labels": Padded tensor of shape [batch_size, aligned_seq_len].

		"""
		input_ids = [torch.as_tensor(item["input_ids"]) for item in batch]
		labels = [torch.as_tensor(item["labels"]) for item in batch]

		padded_input_ids = torch.nn.utils.rnn.pad_sequence(
			input_ids, batch_first=True, padding_value=pad_token_id,
		)

		padded_labels = torch.nn.utils.rnn.pad_sequence(
			labels, batch_first=True, padding_value=ignore_index,
		)

		curr_len = padded_input_ids.shape[1]
		if curr_len % 8 != 0:
			pad_amt = 8 - (curr_len % 8)
			padded_input_ids = torch.nn.functional.pad(
				padded_input_ids, (0, pad_amt), value=pad_token_id,
			)
			padded_labels = torch.nn.functional.pad(
				padded_labels, (0, pad_amt), value=ignore_index,
			)

		return {
			"input_ids": padded_input_ids,
			"labels": padded_labels,
		}

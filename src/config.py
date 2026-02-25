from dataclasses import dataclass
import os
import json
from pathlib import Path
from easy_logging import EasyFormatter
import logging

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger("config.py")
logger.addHandler(handler)


@dataclass
class Config:
	"""Dataclass for MambaCipherSolver configuration.

	Attributes:
		d_model (int, optional): The size of the model. Defaults to 128.
		n_layers (int, optional): The number of layers in the model. Defaults to 4.
		d_state (int, optional): The size of the state. Defaults to 64.
		d_conv (int, optional): The size of the convolution. Defaults to 4.
		expand (int, optional): The expansion factor. Defaults to 2.
		batch_size (int, optional): The batch size. Defaults to 128.
		learning_rate (float, optional): The learning rate. Defaults to 5e-4.
		epochs (int, optional): The number of epochs. Defaults to 30.
		patience (int, optional): The patience for early stopping. Defaults to 1.
		factor (float, optional): The factor for early stopping. Defaults to 0.5.
		save_path (Path, optional): The path to save the model.
			Defaults to Path(__file__).parent.parent.parent / "outputs".
		data_dir (Path, optional): The path to the data directory.
			Defaults to Path(__file__).parent.parent.parent.parent / "Ciphers".
		train_data_dir (Path, optional): The path to the training data directory.
			Defaults to data_dir / "Training".
		valid_data_dir (Path, optional): The path to the validation data directory.
			Defaults to data_dir / "Validation".
		test_data_dir (Path, optional): The path to the test data directory.
			Defaults to data_dir / "Test".
		homophone_file (str, optional): The name of the homophone file.
			Defaults to "metadata.json".
		plain_vocab_size (int, optional): The size of the plain vocabulary.
			Defaults to 26.
		unique_homophones (int, optional): The number of unique homophones.
			Defaults to 500.
		max_len (int, optional): The maximum length of the dataset. Defaults to 1000.
		vocab_size (int, optional): The size of the vocabulary. Defaults to 0.
		buffer (int, optional): The buffer size. Defaults to 1.

	"""

	d_model: int = 128
	n_layers: int = 4
	d_state: int = 64
	d_conv: int = 4
	expand: int = 2

	batch_size: int = 128
	learning_rate: float = 5e-4
	epochs: int = 30

	patience: int = 1
	factor: float = 0.5

	save_path: Path = Path(__file__).parent.parent / "outputs"
	data_dir = Path(__file__).parent.parent.parent / "Ciphers"
	train_data_dir: Path = data_dir / "Training"
	valid_data_dir: Path = data_dir / "Validation"
	test_data_dir: Path = data_dir / "Test"
	homophone_file: str = "metadata.json"

	plain_vocab_size: int = 26
	unique_homophones: int = 3000
	max_len: int = 12_000
	vocab_size: int = 0
	buffer: int = 1

	@classmethod
	def from_homophone_file(cls, file_path: Path) -> "Config":
		"""Create a Config object from a JSON file with max homophone count.

		Args:
			file_path (Path): The path to the homophone count JSON file.

		Returns:
			Config: The Config object.

		"""
		unique_homophones = None
		if os.path.exists(file_path):
			try:
				with open(file_path) as f:
					data = json.load(f)
					unique_homophones = int(
						data.get("max_symbol_id"),
					)
			except (OSError, json.JSONDecodeError, ValueError) as e:
				logger.warning(
					f"Could not parse {file_path}. Using default. Error: {e}",
				)

		if unique_homophones is None:
			return cls()

		return cls(
			unique_homophones=unique_homophones,
			vocab_size=unique_homophones + 26 + 1,
		)

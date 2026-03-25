from dataclasses import dataclass
from pathlib import Path

TEXT_LEN = 9961
TOTAL_SEQ = TEXT_LEN * 2 + 3

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
		save_step (int): Frequency of intermediate checkpoint saves in steps.
			Defaults to 5000.
		patience (int, optional): The patience for early stopping. Defaults to 1.
		factor (float, optional): The factor for early stopping. Defaults to 0.5.
		pct_start (float): Percentage of training for LR warmup. Defaults to 0.1.
		save_path (Path): Directory where model outputs and logs are stored.
		data_dir (Path): Base directory for all cipher-related data.
		tok_train_normal (Path): Path to normal pre-tokenized training data.
		tok_valid_normal (Path): Path to normal pre-tokenized validation data.
		tok_test_normal (Path): Path to normal pre-tokenized test data.
		tok_train_spaced (Path): Path to spaced pre-tokenized training data.
		tok_valid_spaced (Path): Path to spaced pre-tokenized validation data.
		tok_test_spaced (Path): Path to spaced pre-tokenized test data.
		homophone_file (str, optional): The name of the homophone file.
			Defaults to "metadata.json".
		plain_vocab_size (int, optional): The size of the plain vocabulary.
			Defaults to 26.
		unique_homophones (int, optional): The number of unique homophones.
			Defaults to 500.
		max_len (int, optional): The maximum length of the dataset.
		vocab_size (int, optional): The size of the vocabulary. Defaults to 0.
		buffer (int, optional): The buffer size. Defaults to 1.

	"""

	d_model: int = 256
	n_layers: int = 6
	d_state: int = 32
	d_conv: int = 4
	expand: int = 2

	batch_size: int = 8
	learning_rate: float = 5e-4
	epochs: int = 10
	save_step: int = 5000

	patience: int = 2
	factor: float = 0.5
	pct_start: float = 0.1

	save_path: Path = Path(__file__).parent.parent / "outputs"
	data_dir = Path(__file__).parent.parent.parent / "Ciphers"
	tok_train_normal: Path = data_dir / "tokenized_normal" / "Training"
	tok_valid_normal: Path = data_dir / "tokenized_normal" / "Validation"
	tok_test_normal: Path = data_dir / "tokenized_normal" / "Test"
	tok_train_spaced: Path = data_dir / "tokenized_spaced" / "Training"
	tok_valid_spaced: Path = data_dir / "tokenized_spaced" / "Validation"
	tok_test_spaced: Path = data_dir / "tokenized_spaced" / "Test"
	homophone_file: str = "metadata.json"

	plain_vocab_size: int = 26
	unique_homophones: int = 2503
	max_len: int = TOTAL_SEQ
	buffer: int = 50
	pad_token_id = 0

	sep_token_id = unique_homophones + 1
	space_token_id = unique_homophones + 2
	bos_token_id = unique_homophones + 3
	eos_token_id = unique_homophones + 4
	char_offset = unique_homophones + 5

	@property
	def vocab_size(self) -> int:
		"""Calculate the total vocabulary size including offsets and buffer.

		Returns:
			The total integer size of the vocabulary.

		"""
		return self.char_offset + self.plain_vocab_size + self.buffer

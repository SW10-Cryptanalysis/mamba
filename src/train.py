from concurrent.futures import ProcessPoolExecutor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from mamba_ssm import Mamba2
from mamba_ssm.ops.triton.layer_norm import RMSNorm
import os
import json
import zipfile
from tqdm import tqdm
from datetime import datetime
from dataclasses import asdict
from src.config import Config
import logging
from easy_logging import EasyFormatter
from pathlib import Path

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger("train.py")
logger.addHandler(handler)

config = Config()


class CipherDataset(Dataset):
	"""CipherDataset dataset.

	Attributes:
		max_seq_len (int): The maximum sequence length for the dataset.
		file_paths (list[tuple[str, str | None]]): The list of file paths.
		mapping (dict[str, int]): The mapping of characters to IDs.
		sep_token (int): The separator token for the dataset.
		char_offset (int): The offset to add to the character IDs.

	"""

	def __init__(self, directory_path: Path, max_seq_len: int) -> None:
		"""CipherDataset dataset.

		Args:
			directory_path (Path): The path to the directory containing the data.
			max_seq_len (int): The maximum sequence length for the dataset.

		"""
		self.max_seq_len = max_seq_len
		self.file_paths = []

		logger.info(f"Loading file paths from {directory_path}...")
		with os.scandir(directory_path) as entries:
			for entry in tqdm(entries, desc="Scanning for JSON/ZIP", leave=False):
				if entry.is_file():
					if entry.name.endswith(".json"):
						self.file_paths.append((entry.path, None))
					elif entry.name.endswith(".zip"):
						with zipfile.ZipFile(entry.path, "r") as z:
							for file_info in z.infolist():
								if file_info.filename.endswith(".json"):
									self.file_paths.append(
										(entry.path, file_info.filename),
									)

		logger.info(f"Successfully indexed {len(self.file_paths)} files.")
		self.mapping = {chr(i + 97): i for i in range(26)}
		self.sep_token = config.unique_homophones + 1
		self.char_offset = self.sep_token + 1

	def _pad_trunc(self, list_data: list[int]) -> list[int]:
		"""Pad and truncate the list data.

		Args:
			list_data (list[int]): The list of data to pad and truncate.

		Returns:
			list[int]: The padded and truncated list of data.

		"""
		if len(list_data) > self.max_seq_len:
			return list_data[: self.max_seq_len]
		return list_data + [0] * (self.max_seq_len - len(list_data))

	def __len__(self) -> int:
		"""Get the length of the dataset.

		Returns:
			int: The length of the dataset.

		"""
		return len(self.file_paths)

	def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
		"""Get the item at the specified index.

		Args:
			idx (int): The index to get the item at.

		Returns:
			tuple[torch.Tensor, torch.Tensor]: The cipher and plain tensors.

		"""
		path, internal_name = self.file_paths[idx]

		try:
			if internal_name:
				with zipfile.ZipFile(path, "r") as z, z.open(internal_name) as f:
					data = json.load(f)
			else:
				with open(path) as f:
					data = json.load(f)

			ciphertext = data["ciphertext"]
			if isinstance(ciphertext, str):
				ciphertext = [int(x) for x in ciphertext.split()]
			plaintext = data["plaintext"]
			encoded_plain = [
				self.mapping[c] + self.char_offset
				for c in plaintext.lower()
				if c in self.mapping
			]
			cipher_tensor = torch.tensor(self._pad_trunc(ciphertext), dtype=torch.long)
			plain_tensor = torch.tensor(
				self._pad_trunc(encoded_plain),
				dtype=torch.long,
			)
			return cipher_tensor, plain_tensor
		except Exception as e:
			location = f"{path} -> {internal_name}" if internal_name else path
			raise RuntimeError(f"Failed to process {location}: {e}") from e


def process_json(filepath: Path) -> tuple[int, int]:
	"""Process a JSON file and return the length and maximum value.

	Args:
		filepath (Path): The path to the JSON file to process.

	Returns:
		tuple[int, int]: The length and maximum value of the file.

	"""
	try:
		with open(filepath) as f:
			data = json.load(f)

		ciphertext = data.get("ciphertext", [])
		if isinstance(ciphertext, str):
			ciphertext = [int(x) for x in ciphertext.split()]

		actual_max_val = max(ciphertext) if ciphertext else 0
		actual_len = len(ciphertext)

		return actual_len, actual_max_val
	except Exception as e:
		logger.warning(f"Skipping malformed file {filepath}: {e}")
		return None


def get_max_stats(directory_path: str) -> tuple[int, int]:
	"""Scan a directory for the maximum length and maximum symbol ID.

	Args:
		directory_path (str): The path to the directory to scan.

	Returns:
		tuple[int, int]: The maximum length and maximum symbol ID.

	"""
	logger.info(f"Scanning directory: {directory_path}...")

	files = [
		os.path.join(directory_path, f)
		for f in os.listdir(directory_path)
		if f.endswith(".json")
	]

	if not files:
		raise FileNotFoundError(
			f"No JSON files found in the directory: {directory_path}",
		)

	max_length = 0
	max_symbols = 0
	skipped_count = 0

	with ProcessPoolExecutor() as executor:
		results = list(
			tqdm(
				executor.map(process_json, files),
				total=len(files),
				desc="Analyzing Dataset Dimensions",
			),
		)

	for res in results:
		if res is None:
			skipped_count += 1
			continue

		length, symbols = res
		if length > max_length:
			max_length = length
		if symbols > max_symbols:
			max_symbols = symbols

	if skipped_count > 0:
		logger.warning(
			f"\nFinished with warnings: {skipped_count} files were malformed and "
			"skipped.",
		)

	logger.info(
		f"Scan complete. Max Seq Len: {max_length}, Highest Symbol ID: {max_symbols}",
	)
	return max_length, max_symbols


class MambaCipherSolver(nn.Module):
	"""MambaCipherSolver model.

	Attributes:
		char_offset (int): The offset to add to the character IDs.
		embedding (nn.Embedding): The embedding layer for the model.
		layers (nn.ModuleList): The list of layers in the model.
		norm_f (RMSNorm): The final layer norm for the model.
		lm_head (nn.Linear): The linear layer for the model.

	"""

	def __init__(
		self,
		vocab_size: int,
		char_offset: int,
		d_model: int = 128,
		n_layers: int = 4,
	) -> None:
		"""MambaCipherSolver model.

		Args:
			vocab_size (int): The size of the vocabulary for the ciphers.
			char_offset (int): The offset to add to the character IDs.
			d_model (int, optional): The size of the model. Defaults to 128.
			n_layers (int, optional): The number of layers in the model. Defaults to 4.

		"""
		super().__init__()
		self.char_offset = char_offset
		self.embedding = nn.Embedding(vocab_size, d_model)

		self.layers = nn.ModuleList(
			[
				nn.ModuleDict(
					{
						"norm": RMSNorm(d_model),
						"mixer": Mamba2(
							d_model=d_model,
							d_state=config.d_state,
							d_conv=config.d_conv,
							expand=config.expand,
						),
					},
				)
				for _ in range(n_layers)
			],
		)

		self.norm_f = RMSNorm(d_model)
		self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Forward pass of the model.

		Args:
			x (torch.Tensor): The input tensor to the model.

		Returns:
			torch.Tensor: The output tensor of the model.

		"""
		x = self.embedding(x)
		for layer in self.layers:
			residual = x
			x = layer["norm"](x)
			x = layer["mixer"](x) + residual

		x = self.norm_f(x)
		return self.lm_head(x)


def train_model(
	model: MambaCipherSolver,
	train_loader: DataLoader,
	val_loader: DataLoader,
	cipher_vocab: int,
	epochs: int = 10,
	save_path: Path | None = None,
) -> None:
	"""Trains a MambaCipherSolver model on the provided dataloaders.

	Args:
		model (MambaCipherSolver): The model to train.
		train_loader (DataLoader): The dataloader to use for training.
		val_loader (DataLoader): The dataloader to use for validation.
		cipher_vocab (int): The size of the vocabulary for the ciphers.
		epochs (int, optional): The number of epochs to train for. Defaults to 10.
		save_path (Path | None, optional): The path to save the model to.
			Defaults to None.

	"""
	if save_path is None:
		save_path = Path(__file__).parent.parent.parent / "outputs"
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

	scheduler = optim.lr_scheduler.ReduceLROnPlateau(
		optimizer,
		mode="min",
		factor=config.factor,
		patience=config.patience,
	)

	best_val_loss = float("inf")

	for epoch in range(epochs):
		model.train()
		total_train_loss = 0
		train_loop = tqdm(
			train_loader,
			desc=f"Epoch {epoch + 1}/{epochs} [Train]",
			unit="batch",
			leave=False,
		)

		for cipher, plain in train_loop:
			cipher, plain = cipher.to("cuda"), plain.to("cuda")
			optimizer.zero_grad()
			outputs = model(cipher)
			loss = criterion(outputs.view(-1, outputs.size(-1)), plain.view(-1))
			loss.backward()
			optimizer.step()
			total_train_loss += loss.item()

		model.eval()
		total_val_loss = 0
		with torch.no_grad():
			for cipher, plain in val_loader:
				cipher, plain = cipher.to("cuda"), plain.to("cuda")
				outputs = model(cipher)
				loss = criterion(outputs.view(-1, outputs.size(-1)), plain.view(-1))
				total_val_loss += loss.item()

		avg_train_loss = total_train_loss / len(train_loader)
		avg_val_loss = total_val_loss / len(val_loader)

		current_lr = optimizer.param_groups[0]["lr"]
		scheduler.step(avg_val_loss)

		logger.info(
			f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_train_loss:.4f} "
			"| Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}",
		)

		if avg_val_loss < best_val_loss:
			best_val_loss = avg_val_loss
			torch.save(
				{
					"model_state_dict": model.state_dict(),
					"val_loss": avg_val_loss,
					"cipher_vocab": cipher_vocab,
					"char_offset": model.char_offset,
				},
				save_path,
			)

		if current_lr < 1e-7:
			logger.info(f"Current lr: {current_lr}. Stopping early.")
			break


if __name__ == "__main__":
	train_path = os.path.abspath(config.train_data_dir)
	valid_path = os.path.abspath(config.valid_data_dir)

	if isinstance(config.unique_homophones, int) and isinstance(config.max_len, int):
		max_len = config.max_len
		cipher_vocab = config.unique_homophones
	else:
		max_len, cipher_vocab = get_max_stats(train_path)
		if max_len == 0:
			raise ValueError(f"No valid JSON data found in {train_path}.")

	cores = os.cpu_count() or 1
	num_workers = max(1, cores - 4)

	train_dataset = CipherDataset(train_path, max_seq_len=max_len)
	train_loader = DataLoader(
		train_dataset,
		batch_size=config.batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=True,
		persistent_workers=True,
	)

	val_dataset = CipherDataset(valid_path, max_seq_len=max_len)
	val_loader = DataLoader(
		val_dataset,
		batch_size=config.batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=True,
		persistent_workers=True,
	)

	vocab_size = train_dataset.char_offset + config.plain_vocab_size + config.buffer

	model = MambaCipherSolver(
		vocab_size=vocab_size,
		char_offset=train_dataset.char_offset,
		d_model=config.d_model,
		n_layers=config.n_layers,
	).to("cuda")

	timestamp = datetime.now().strftime("%m%d-%H%M")
	os.makedirs(config.save_path, exist_ok=True)
	config_filename = os.path.join(config.save_path, f"config_{timestamp}.json")
	with open(config_filename, "w") as f:
		json.dump(asdict(config), f, indent=4)
	filename = f"mamba2_{timestamp}.pth"

	logger.info("Training...")
	train_model(
		model,
		train_loader,
		val_loader,
		cipher_vocab + config.buffer,
		epochs=config.epochs,
		save_path=Path(config.save_path) / filename,
	)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from mamba_ssm import Mamba2
from mamba_ssm.ops.triton.layer_norm import RMSNorm
import os
import json
from tqdm import tqdm
from datetime import datetime
from dataclasses import asdict
from src.config import Config
from src.utils.data_manager import DatasetManager
from src.data.dataset import CipherDataset
import logging
from easy_logging import EasyFormatter
from pathlib import Path

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger("train.py")
logger.addHandler(handler)

config = Config()

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
					"d_model": config.d_model,
    				"n_layers": config.n_layers
				},
				save_path,
			)

		if current_lr < 1e-7:
			logger.info(f"Current lr: {current_lr}. Stopping early.")
			break

if __name__ == "__main__":
	train_path = os.path.abspath(config.train_data_dir)
	valid_path = os.path.abspath(config.valid_data_dir)

	train_file_list = DatasetManager.scan_directory(train_path)
	valid_file_list = DatasetManager.scan_directory(valid_path)

	if isinstance(config.unique_homophones, int) and isinstance(config.max_len, int):
		max_len = config.max_len
		cipher_vocab = config.unique_homophones
	else:
		max_len, cipher_vocab = DatasetManager.get_max_stats(train_file_list)
		if max_len == 0:
			raise ValueError(f"No valid JSON data found in {train_path}.")

	cores = os.cpu_count() or 1
	num_workers = max(1, cores - 4)

	train_dataset = CipherDataset(train_file_list, max_seq_len=max_len, config=config)
	train_loader = DataLoader(
		train_dataset,
		batch_size=config.batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=True,
		persistent_workers=True,
	)

	val_dataset = CipherDataset(valid_file_list, max_seq_len=max_len, config=config)
	val_loader = DataLoader(
		val_dataset,
		batch_size=config.batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=True,
		persistent_workers=True,
	)

	vocab_size = train_dataset.char_offset + config.plain_vocab_size + config.buffer
	print(f"--- MEMORY DEBUG ---")
	print(f"Max Sequence Length: {max_len}")
	print(f"Vocab Size: {vocab_size}")
	print(f"Batch Size: {config.batch_size}")
	print(f"Model Dimension (d_model): {config.d_model}")
	print(f"---------------------")
	model = MambaCipherSolver(
		vocab_size=vocab_size,
		char_offset=train_dataset.char_offset,
		d_model=config.d_model,
		n_layers=config.n_layers,
	).to("cuda")

	timestamp = datetime.now().strftime("%d%m-%H%M")
	os.makedirs(config.save_path, exist_ok=True)
	config_filename = os.path.join(config.save_path, f"config_{timestamp}.json")
	with open(config_filename, "w") as f:
		json.dump(asdict(config), f, indent=4, default=str)
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

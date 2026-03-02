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
from src.data.tokenizer import CipherTokenizer
from src.engine.trainer import MambaTrainer
import logging
from easy_logging import EasyFormatter
from pathlib import Path

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger("train.py")
logger.addHandler(handler)

config = Config()
tokenizer = CipherTokenizer(config)

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

	train_dataset = CipherDataset(train_file_list, max_seq_len=max_len, tokenizer=tokenizer)
	train_loader = DataLoader(
		train_dataset,
		batch_size=config.batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=True,
		persistent_workers=True,
	)

	val_dataset = CipherDataset(valid_file_list, max_seq_len=max_len, tokenizer=tokenizer)
	val_loader = DataLoader(
		val_dataset,
		batch_size=config.batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=True,
		persistent_workers=True,
	)

	vocab_size = tokenizer.char_offset + config.plain_vocab_size + config.buffer
	
	model = MambaCipherSolver(
		vocab_size=tokenizer.vocab_size,
		char_offset=tokenizer.char_offset,
		d_model=config.d_model,
		n_layers=config.n_layers,
	).to("cuda")

	timestamp = datetime.now().strftime("%d%m-%H%M")
	os.makedirs(config.save_path, exist_ok=True)
	checkpoint_file = Path(config.save_path) / f"mamba2_{timestamp}.pth"
	model_config = Path(config.save_path) / f"config_{timestamp}.json"
	with open(model_config, "w") as f:
		json.dump(asdict(config), f, indent=4, default=str)

	trainer = MambaTrainer(
		model=model,
		train_loader=train_loader,
		val_loader=val_loader,
		config=config,
		save_path=checkpoint_file
	)

	logger.info("Training...")
	trainer.train(epochs=config.epochs)
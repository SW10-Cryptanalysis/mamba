import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mamba_ssm import Mamba2
from mamba_ssm.ops.triton.layer_norm import RMSNorm
import os
import argparse
from src.utils.logging import get_logger
from pathlib import Path
from src.config import Config
from src.utils.data_manager import DataManager
from src.data.dataset import CipherDataset
from src.data.tokenizer import CipherTokenizer
from src.engine.trainer import MambaTrainer

logger = get_logger("train.py")

class MambaModel(nn.Module):
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
		config: Config
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
		self.embedding = nn.Embedding(vocab_size, config.d_model)

		self.layers = nn.ModuleList(
			[
				nn.ModuleDict(
					{
						"norm": RMSNorm(config.d_model),
						"mixer": Mamba2(
							d_model=config.d_model,
							d_state=config.d_state,
							d_conv=config.d_conv,
							expand=config.expand,
						),
					},
				)
				for _ in range(config.n_layers)
			],
		)

		self.norm_f = RMSNorm(config.d_model)
		self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)

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
	config = Config()
	parser = argparse.ArgumentParser()
	parser.add_argument("--resume", nargs="?", const="auto", default=None)
	args = parser.parse_args()

	save_path = Path(config.save_path)
	resume_path = None
	target_exp_dir = None
	
	if args.resume == "auto":
		resume_path = DataManager.get_latest_checkpoint(save_path)
		if resume_path:
			target_exp_dir = resume_path.parent
			logger.info(f"Auto-detected checkpoint: {resume_path}")
	elif args.resume:
		resume_path = Path(args.resume)
		if resume_path.exists():
			target_exp_dir = resume_path.parent
		else:
			raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

	if target_exp_dir:
		config_json = target_exp_dir / "config.json"
		if config_json.exists():
			logger.info(f"Resuming training: replacing global config with {config_json}")
			with open(config_json, "r") as f:
				config_dict = json.load(f)
				for key, value in config_dict.items():
					if hasattr(config, key):
						if isinstance(value, str) and ("path" in key or "dir" in key):
							value = Path(value)
						setattr(config, key, value)
		else:
			logger.warning("No config.json found in resume folder. Using current global settings.")

	tokenizer = CipherTokenizer(config)

	train_path = os.path.abspath(config.train_data_dir)
	valid_path = os.path.abspath(config.valid_data_dir)
	train_file_list = DataManager.scan_directory(train_path)
	valid_file_list = DataManager.scan_directory(valid_path)
	
	if isinstance(config.unique_homophones, int) and isinstance(config.max_len, int):
		max_len = config.max_len
		cipher_vocab = config.unique_homophones
	else:
		max_len, cipher_vocab = DataManager.get_max_stats(train_file_list)
		if max_len == 0:
			raise ValueError(f"No valid JSON data found in {train_path}.")

	cores = os.cpu_count() or 1
	num_workers = max(1, cores - 4)

	train_loader = DataLoader(
		CipherDataset(train_file_list, max_seq_len=max_len, tokenizer=tokenizer),
		batch_size=config.batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=True,
		persistent_workers=True,
	)

	val_loader = DataLoader(
		CipherDataset(valid_file_list, max_seq_len=max_len, tokenizer=tokenizer),
		batch_size=config.batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=True,
		persistent_workers=True,
	)

	model = MambaModel(
		vocab_size=tokenizer.vocab_size,
		char_offset=tokenizer.char_offset,
		config=config
	).to("cuda")

	trainer = MambaTrainer(
		model=model,
		train_loader=train_loader,
		val_loader=val_loader,
		config=config,
		save_path=save_path,
		exp_dir=target_exp_dir
	)

	if resume_path:
		trainer.load_checkpoint(resume_path)
	
	trainer.train(epochs=config.epochs)
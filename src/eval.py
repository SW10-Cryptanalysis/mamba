import glob
import os
import json
import torch
import argparse
import zipfile
from src.train import MambaCipherSolver
from src.config import Config
from easy_logging import EasyFormatter
import logging
from pathlib import Path

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger("eval.py")
logger.addHandler(handler)

config = Config()


def decode_plain(indices: list[int], char_offset: int) -> str:
	"""Convert list of integers (0-25) back into a string (a-z).

	Args:
		indices (list[int]): The list of integers to decode.
		char_offset (int): The offset to add to the character IDs.

	Returns:
		str: The decoded string.

	"""
	mapping = {i + char_offset: chr(i + 97) for i in range(26)}
	return "".join([mapping.get(idx, "?") for idx in indices])


def ser(pred: str, plaintext: str) -> float:
	"""Calculate the symbol error rate (SER) for a predicted and plaintext string.

	Args:
		pred (str): The predicted string.
		plaintext (str): The plaintext string.

	Returns:
		float: The symbol error rate (SER).

	"""
	if len(plaintext) == 0:
		return 0.0
	count = sum(1 for p, t in zip(pred, plaintext, strict=True) if p == t)
	return 1.0 - (count / len(plaintext))


def test_model(test_dir: Path, model_path: Path | None = None) -> None: # noqa: C901
	"""Test a MambaCipherSolver model on a directory of JSON files.

	Args:
		test_dir (Path): The path to the directory containing the JSON files.
		model_path (Path | None, optional): The path to the model to test.
			Defaults to None.

	"""
	if model_path is None:
		list_of_files = glob.glob(os.path.join(config.save_path, "*.pth"))
		if not list_of_files:
			logger.error(f"No models found in {config.save_path}")
			return None
		model_path = Path(max(list_of_files, key=os.path.getctime))

	if not os.path.exists(model_path):
		alternative_path = config.save_path / model_path
		if os.path.exists(alternative_path):
			model_path = alternative_path
		else:
			logger.error(f"Could not find model at {model_path}")
			return None

	logger.info(f"Testing model: {model_path}")

	checkpoint = torch.load(model_path, map_location="cuda")

	char_offset = checkpoint["char_offset"]
	vocab_size = char_offset + config.plain_vocab_size + config.buffer

	model = MambaCipherSolver(vocab_size, char_offset).to("cuda")
	model.load_state_dict(checkpoint["model_state_dict"])
	model.eval()
	results = {
		"model_path": model_path,
		"predictions": [],
	}

	if not os.path.exists(test_dir):
		logger.error(f"{test_dir} not found.")
		return

	test_samples = []
	with os.scandir(test_dir) as entries:
		for entry in entries:
			if entry.is_file():
				if entry.name.endswith(".json"):
					test_samples.append((entry.path, None))
				elif entry.name.endswith(".zip"):
					with zipfile.ZipFile(entry.path, "r") as z:
						for name in z.namelist():
							if name.endswith(".json"):
								test_samples.append((entry.path, name))

	logger.info(f"Testing {len(test_samples)} files...")

	with torch.no_grad():
		for path, internal_name in test_samples:
			if internal_name:
				with zipfile.ZipFile(path, "r") as z, z.open(internal_name) as f:
					data = json.load(f)
				filename = f"{os.path.basename(path)}/{internal_name}"
			else:
				with open(path) as f:
					data = json.load(f)
				filename = os.path.basename(path)

			raw_cipher = data["ciphertext"]
			if isinstance(raw_cipher, str):
				raw_cipher = [int(x) for x in raw_cipher.split()]

			input_tensor = (
				torch.tensor(raw_cipher, dtype=torch.long).unsqueeze(0).to("cuda")
			)

			logits = model(input_tensor)
			pred_indices = torch.argmax(logits, dim=-1).squeeze().tolist()

			if isinstance(pred_indices, int):
				pred_indices = [pred_indices]

			deciphered_text = decode_plain(pred_indices, char_offset)
			plaintext = data["plaintext"]
			symbol_err_rate = ser(deciphered_text, plaintext)

			results["predictions"].append(
				{
					"filename": filename,
					"ground_truth": plaintext,
					"predicted": deciphered_text,
					"ser": symbol_err_rate,
				},
			)
			logger.info(
				f"File: {filename} | SER: {symbol_err_rate} | "
				"Predicted: {deciphered_text}",
			)

	full_model_name = os.path.splitext(os.path.basename(model_path))[0]
	if "_" in full_model_name:
		identifier = full_model_name.split("_")[-1]
	else:
		identifier = full_model_name
	output_filename = os.path.join(config.save_path, f"eval_{identifier}.json")
	with open(output_filename, "w") as f:
		json.dump(results, f, indent=4)
	logger.info(f"\nResults saved to {output_filename}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Evaluate a Mamba Cipher Model")
	parser.add_argument(
		"model_name",
		nargs="?",
		default=None,
		help="Name of the .pth model file (defaults to latest in output dir)",
	)
	args = parser.parse_args()

	m_path = (
		os.path.join(config.save_path, args.model_name) if args.model_name else None
	)

	test_model(config.test_data_dir, model_path=m_path)

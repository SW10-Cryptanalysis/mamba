import glob
import os
import json
import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from src.train import MambaCipherSolver
from src.config import Config
from src.utils.data_manager import DatasetManager
from src.data.dataset import CipherDataset
from easy_logging import EasyFormatter
import logging

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
    if not plaintext:
        return 0.0
    count = sum(1 for p, t in zip(pred, plaintext) if p == t)
    return 1.0 - (count / len(plaintext))

def test_model(test_dir: Path, model_path: Path | None = None) -> None:
    if model_path is None:
        list_of_files = glob.glob(os.path.join(config.save_path, "*.pth"))
        if not list_of_files:
            logger.error(f"No models found in {config.save_path}")
            return
        model_path = Path(max(list_of_files, key=os.path.getctime))

    if not model_path.exists():
        model_path = config.save_path / model_path.name
        if not model_path.exists():
            logger.error(f"Could not find model at {model_path}")
            return

    logger.info(f"Testing model: {model_path}")

    checkpoint = torch.load(model_path, map_location="cuda", weights_only=False)
    char_offset = checkpoint["char_offset"]
    d_model = checkpoint.get("d_model", config.d_model)
    n_layers = checkpoint.get("n_layers", config.n_layers)
    vocab_size = char_offset + config.plain_vocab_size + config.buffer

    logger.info(f"Reconstructing model: d_model={d_model}, n_layers={n_layers}, vocab={vocab_size}")

    model = MambaCipherSolver(
        vocab_size=vocab_size, 
        char_offset=char_offset,
        d_model=d_model, 
        n_layers=n_layers
    ).to("cuda")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_files = DatasetManager.scan_directory(test_dir)
    test_dataset = CipherDataset(test_files, max_seq_len=config.max_len, config=config, mode="eval")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    results = {
        "model_path": str(model_path),
        "predictions": [],
    }

    logger.info(f"Testing {len(test_dataset)} files...")
    
    with torch.no_grad():
        for cipher_tensor, ground_truth, metadata in test_loader:
            cipher_tensor = cipher_tensor.to("cuda")
            
            logits = model(cipher_tensor)
            pred_indices = torch.argmax(logits, dim=-1).squeeze(0).tolist()

            current_ground_truth = ground_truth[0]
            current_path = metadata['path'][0]
            current_internal = metadata['internal_name'][0] if metadata['internal_name'][0] else ""
            
            filename = f"{os.path.basename(current_path)}/{current_internal}" if current_internal else os.path.basename(current_path)
            
            deciphered_text = decode_plain(pred_indices, char_offset)
            deciphered_text = deciphered_text[:len(current_ground_truth)]
            
            symbol_err_rate = ser(deciphered_text, current_ground_truth)

            results["predictions"].append({
                "filename": filename,
                "ground_truth": current_ground_truth,
                "predicted": deciphered_text,
                "ser": symbol_err_rate,
            })

            logger.info(f"File: {filename} | SER: {symbol_err_rate:.4f} | Predicted: {deciphered_text}")

    full_model_name = model_path.stem
    identifier = full_model_name.split("_")[-1] if "_" in full_model_name else full_model_name
    output_filename = os.path.join(config.save_path, f"eval_{identifier}.json")
    
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"\nResults saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Mamba Cipher Model")
    parser.add_argument("model_name", nargs="?", default=None)
    args = parser.parse_args()

    m_path = Path(os.path.join(config.save_path, args.model_name)) if args.model_name else None
    test_model(Path(config.test_data_dir), model_path=m_path)
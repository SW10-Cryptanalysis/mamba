import os
import json
import glob
import argparse
import logging
from pathlib import Path
from torch.utils.data import DataLoader
from easy_logging import EasyFormatter
from src.config import Config
from src.utils.data_manager import DataManager
from src.data.dataset import CipherDataset
from src.engine.solver import CipherSolver

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger("eval.py")
logger.addHandler(handler)

config = Config()

def test_model(test_dir: Path, model_path: Path | None = None) -> None:
    if model_path is None:
        list_of_files = glob.glob(os.path.join(config.save_path, "*.pth"))
        if not list_of_files:
            logger.error(f"No models found in {config.save_path}")
            return
        model_path = Path(max(list_of_files, key=os.path.getctime))

    solver = CipherSolver(config)
    solver.load_checkpoint(model_path)

    test_files = DataManager.scan_directory(test_dir)
    test_dataset = CipherDataset(test_files, max_seq_len=config.max_len, tokenizer=solver.tokenizer, mode="eval")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    results = {"model_path": str(model_path), "predictions": []}

    logger.info(f"Testing {len(test_dataset)} files with solver...")

    for cipher_tensor, ground_truth, metadata in test_loader:
        raw_cipher = cipher_tensor.squeeze(0).tolist()
        
        current_ground_truth = ground_truth[0]
        current_path = metadata['path'][0]
        current_internal = metadata['internal_name'][0] if metadata['internal_name'][0] else ""
        filename = f"{os.path.basename(current_path)}/{current_internal}" if current_internal else os.path.basename(current_path)

        deciphered_text = solver.decrypt(raw_cipher)
        symbol_err_rate = solver.calculate_ser(deciphered_text, current_ground_truth)

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
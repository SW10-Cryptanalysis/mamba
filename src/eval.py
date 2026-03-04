import os
import json
import glob
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from src.config import Config
from src.utils.data_manager import DataManager
from src.data.dataset import CipherDataset
from src.engine.solver import CipherSolver
from src.utils.logging import get_logger
logger = get_logger("eval.py")

def test_model(test_dir: Path, model_path: Path | None = None) -> None:
    """Evaluate a cipher model on a test directory and save the results.

    If no model path is provided, the function automatically searches the
    configuration's save directory for the most recently modified checkpoint.
    It processes the test data in 'eval' mode, calculates the Symbol Error
    Rate (SER) for each sample, and exports the final predictions to a JSON file.

    Args:
        test_dir: Path to the directory containing test ciphertext samples.
        model_path: Optional path to a specific model checkpoint. If None,
            the latest model in config.save_path is used.

    Returns:
        None. Results are saved to a file named 'eval_{model_name}.json'
        in the model's parent directory.

    """
    if model_path is None:
        list_of_files = glob.glob(
            os.path.join(config.save_path, "**/*.pth"),
            recursive=True,
        )
        if not list_of_files:
            logger.error(f"No models found in {config.save_path}")
            return
        model_path = Path(max(list_of_files, key=os.path.getmtime))

    model_dir = model_path.parent
    logger.info(f"Eval on model: {model_path}")

    solver = CipherSolver(config)
    solver.load_checkpoint(model_path)

    test_files = DataManager.scan_directory(test_dir)
    test_dataset = CipherDataset(
        test_files,
        max_seq_len=config.max_len,
        tokenizer=solver.tokenizer,
        mode="eval",
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    results = {"model_path": str(model_path), "predictions": []}

    logger.info(f"Testing {len(test_dataset)} files with solver...")

    for cipher_tensor, ground_truth, metadata in test_loader:
        raw_cipher = cipher_tensor.squeeze(0).tolist()

        current_ground_truth = ground_truth[0]
        current_path = metadata["path"][0]
        current_internal = metadata["internal_name"][0] or ""
        base = os.path.basename(current_path)
        filename = f"{base}/{current_internal}" if current_internal else base

        deciphered_text = solver.decrypt(raw_cipher)
        symbol_err_rate = solver.calculate_ser(deciphered_text, current_ground_truth)

        results["predictions"].append({
            "filename": filename,
            "ground_truth": current_ground_truth,
            "predicted": deciphered_text,
            "ser": symbol_err_rate,
        })

    full_model_name = model_path.stem
    output_filename = model_dir / f"eval_{full_model_name}.json"

    with open(output_filename, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"\nResults saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Mamba Cipher Model")
    parser.add_argument(
        "model_path",
        nargs="?",
        default=None,
        help="Path to specific .pth file",
    )
    args = parser.parse_args()

    config = Config()

    model_path = None
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        latest_checkpoint = DataManager.get_latest_checkpoint(config.save_path)
        if latest_checkpoint:
            model_path = latest_checkpoint
            logger.info(f"Auto-detected latest model: {model_path}")

    if not model_path or not model_path.exists():
        logger.error("No valid model path provided or found.")
        exit(1)

    exp_dir = model_path.parent
    config_json = exp_dir / "config.json"
    if config_json.exists():
        logger.info(f"Loading experiment config from {config_json}")
        with open(config_json) as f:
            saved_dict = json.load(f)
            for k, v in saved_dict.items():
                if hasattr(config, k):
                    setattr(config, k, v)

    test_model(Path(config.test_data_dir), model_path=model_path)

import os
import json
import glob
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.config import Config
from src.utils.data_manager import DataManager
from src.data.dataset import CipherDataset, PretokenizedCipherDataset
from src.engine.solver import CipherSolver
from src.utils.logging import get_logger
logger = get_logger("eval.py")

def _find_model_path(config_save_path: str) -> Path | None:
    """Find the most recent best.pth or latest .pth file.

    This function searches recursively through the save directory. It prioritizes
    files named 'best.pth' by returning the most recently modified one. If no
    'best.pth' exists, it returns the most recently modified '.pth' file found.

    Args:
        config_save_path: The root directory string where model checkpoints
            are stored.

    Returns:
        The Path to the selected checkpoint file, or None if no .pth files
        are found in the directory.

    """
    list_of_files = glob.glob(
        os.path.join(config_save_path, "**/*.pth"),
        recursive=True,
    )
    if not list_of_files:
        return None

    best_models = [f for f in list_of_files if os.path.basename(f) == "best.pth"]
    if best_models:
        return Path(max(best_models, key=os.path.getmtime))
    return Path(max(list_of_files, key=os.path.getmtime))

def _save_eval_summary(
    model_dir: Path,
    model_stem: str,
    scores: list,
    path: Path,
) -> None:
    """Calculate evaluation statistics and export the final summary JSON.

    Args:
        model_dir: The directory where the summary file will be saved.
        model_stem: The filename stem of the model (e.g., 'best' or 'epoch_001')
            used to name the summary file.
        scores: A list of floats representing the Symbol Error Rate for
            each processed sample.
        path: The full path to the model checkpoint being evaluated,
            included in the summary for traceability.

    """
    if not scores:
        return

    avg_ser = sum(scores) / len(scores)
    summary = {
        "model_path": str(path),
        "average_ser": round(avg_ser, 4),
        "total_samples": len(scores),
        "best_ser": round(min(scores), 4),
        "worst_ser": round(max(scores), 4),
    }

    summary_path = model_dir / f"summary_{model_stem}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    logger.info("--- EVALUATION SUMMARY ---")
    logger.info(f"Final Average SER: {avg_ser:.4f}")
    logger.info(f"Results saved to: {summary_path}")

def test_model(test_dir: Path, model_path: Path | None = None) -> None:
    """Evaluate a cipher model on a test directory and save the results.

    Args:
        test_dir: Path to the directory containing test samples. Can be
            an Arrow-formatted dataset directory or a directory of raw files.
        model_path: Optional path to a specific '.pth' checkpoint. If None,
            the function will attempt to auto-detect the best or latest
            model in the configured save directory.

    Returns:
        None. Per-sample predictions are saved to 'eval_{model_stem}.jsonl'
        and aggregate statistics are saved to 'summary_{model_stem}.json'
        in the model's directory.

    """
    if model_path is None:
        model_path = _find_model_path(config.save_path)
        if not model_path:
            logger.error(f"No models found in {config.save_path}")
            return
        logger.info(f"Using auto-detected model: {model_path}")

    model_dir = model_path.parent
    solver = CipherSolver(config)
    solver.load_checkpoint(model_path)

    test_dir = Path(test_dir).resolve()
    is_arrow = (
        (test_dir / "dataset_info.json").exists()
        or any(test_dir.glob("*.arrow"))
    )

    if is_arrow:
        test_dataset = PretokenizedCipherDataset(test_dir, config.max_len, config)
    else:
        test_files = DataManager.scan_directory(test_dir)
        test_dataset = CipherDataset(
            test_files,
            config.max_len,
            solver.tokenizer,
            "eval",
        )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    output_filename = model_dir / f"eval_{model_path.stem}.jsonl"
    all_ser_scores = []

    with open(output_filename, "a", encoding="utf-8") as f_out:
        pbar = tqdm(test_loader, desc="Decrypting")
        for i, batch in enumerate(pbar):
            try:
                input_ids = batch["input_ids"].squeeze(0).tolist()
                raw_labels = batch["labels"].squeeze(0).tolist()
                label_ids = [tid for tid in raw_labels if tid != -100]
                current_ground_truth = solver.tokenizer.decode(label_ids)

                sep_id = solver.tokenizer.sep_token_id
                if sep_id not in input_ids:
                    continue

                sep_idx = input_ids.index(sep_id)
                deciphered_text = solver.decrypt(input_ids[:sep_idx + 1])
                ser = solver.calculate_ser(deciphered_text, current_ground_truth)

                all_ser_scores.append(ser)
                avg_ser = sum(all_ser_scores) / len(all_ser_scores)
                pbar.set_postfix({"avg_ser": f"{avg_ser:.4f}"})

                result = {
                    "id": batch.get("id", [f"idx_{i}"])[0],
                    "ser": round(ser, 4),
                    "predicted": deciphered_text,
                    "ground_truth": current_ground_truth,
                }
                f_out.write(json.dumps(result) + "\n")
                f_out.flush()

            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")

    _save_eval_summary(model_dir, model_path.stem, all_ser_scores, model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Mamba Cipher Model")
    parser.add_argument(
        "model_path",
        nargs="?",
        default=None,
        help="Path to specific .pth file",
    )
    parser.add_argument(
        "--spaces",
        action="store_true",
        help="Evaluate using the spaced model/dataset",
    )
    args = parser.parse_args()

    config = Config()
    run_type = "spaced" if args.spaces else "normal"
    search_prefix = f"exp_{run_type}_*"

    model_path = None
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        latest_checkpoint = DataManager.get_latest_checkpoint(
            config.save_path,
            prefix=search_prefix,
        )
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

    test_dir = config.tok_test_spaced if args.spaces else config.tok_test_normal
    test_model(test_dir=test_dir, model_path=model_path)

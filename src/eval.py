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

    test_dir = Path(test_dir).resolve()
    is_arrow = (test_dir / "dataset_info.json").exists() or any(test_dir.glob("*.arrow"))

    if is_arrow:
        logger.info(f"Arrow format detected at {test_dir}. Using PretokenizedCipherDataset.")
        test_dataset = PretokenizedCipherDataset(test_dir, max_seq_len=config.max_len, config=config)
    else:
        logger.info("Legacy format detected. Scanning directory...")
        test_files = DataManager.scan_directory(test_dir)
        test_dataset = CipherDataset(
            test_files,
            max_seq_len=config.max_len,
            tokenizer=solver.tokenizer,
            mode="eval",
        )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    logger.info(f"Testing {len(test_dataset)} files with solver...")
    output_filename = model_dir / f"eval_{model_path.stem}.jsonl"

    all_ser_scores = []
    running_ser = 0.0
    count = 0
    pbar = tqdm(test_loader, desc="Decrypting")

    with open(output_filename, "a", encoding="utf-8") as f_out:
        logger.info(f"Testing {len(test_dataset)} files. Results: {output_filename}")

        for i, batch in enumerate(pbar):
            try:
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"].squeeze(0).tolist()
                    label_ids = batch["labels"].squeeze(0).tolist()

                    label_ids = [tid for tid in label_ids if tid != -100]
                    current_ground_truth = solver.tokenizer.decode(label_ids)

                    sep_id = solver.tokenizer.sep_token_id
                    if sep_id in input_ids:
                        sep_idx = input_ids.index(sep_id)
                        raw_cipher = input_ids[:sep_idx + 1]
                    else:
                        logger.warning(f"Sample {i} missing SEP token. Skipping.")
                        continue

                    filename = batch.get("id", [f"sample_{i:06d}"])[0]

                deciphered_text = solver.decrypt(raw_cipher)
                ser = solver.calculate_ser(deciphered_text, current_ground_truth)

                all_ser_scores.append(ser)
                running_ser += ser
                count += 1

                if i % 10 == 0:
                    pbar.set_postfix({"avg_ser": f"{running_ser/count:.4f}"})

                result_entry = {
                    "id": str(filename),
                    "ser": round(ser, 4),
                    "predicted": deciphered_text,
                    "ground_truth": current_ground_truth
                }
                f_out.write(json.dumps(result_entry) + "\n")
                f_out.flush()

            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                continue
    
    if all_ser_scores:
        avg_ser = sum(all_ser_scores) / len(all_ser_scores)
        summary = {
            "model_path": str(model_path),
            "average_ser": round(avg_ser, 4),
            "total_samples": len(all_ser_scores),
            "best_ser": round(min(all_ser_scores), 4),
            "worst_ser": round(max(all_ser_scores), 4)
        }
        summary_path = model_dir / f"summary_{model_path.stem}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"--- EVALUATION SUMMARY ---")
        logger.info(f"Final Average SER: {avg_ser:.4f}")
        logger.info(f"Results saved to: {summary_path}")

    logger.info("Evaluation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Mamba Cipher Model")
    parser.add_argument("model_path", nargs="?", default=None, help="Path to specific .pth file")
    parser.add_argument("--spaces", action="store_true", help="Evaluate using the spaced model/dataset")
    args = parser.parse_args()

    config = Config()
    run_type = "spaced" if args.spaces else "normal"
    search_prefix = f"exp_{run_type}_*"

    model_path = None
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        latest_checkpoint = DataManager.get_latest_checkpoint(config.save_path, prefix=search_prefix)
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

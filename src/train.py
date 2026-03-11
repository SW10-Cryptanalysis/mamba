import json
import os
import argparse
from pathlib import Path
from functools import partial
import torch
from torch.utils.data import DataLoader
from src.models.mamba import MambaModel
from src.utils.logging import get_logger
from src.config import Config
from src.utils.data_manager import DataManager
from src.data.dataset import CipherDataset, PretokenizedCipherDataset
from src.data.tokenizer import CipherTokenizer
from src.engine.trainer import MambaTrainer

logger = get_logger("train.py")

def resolve_config(resume_arg: str | None, config: Config, run_type: str) -> tuple[Path | None, Path | None]:
    """Handle checkpoint auto-detection and sync config from existing experiments."""
    save_path = Path(config.save_path)
    resume_path = None
    target_exp_dir = None

    if resume_arg == "auto":
        search_prefix = f"exp_{run_type}_*"
        resume_path = DataManager.get_latest_checkpoint(save_path, prefix=search_prefix)
    elif resume_arg:
        resume_path = Path(resume_arg)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

    if resume_path:
        target_exp_dir = resume_path.parent
        logger.info(f"Using checkpoint: {resume_path}")

        config_json = target_exp_dir / "config.json"
        if config_json.exists():
            logger.info(f"Resuming: syncing architecture from {config_json}")
            with open(config_json) as f:
                config_dict = json.load(f)
                for key, value in config_dict.items():
                    if hasattr(config, key):
                        if isinstance(value, str) and ("path" in key or "dir" in key):
                            value = Path(value)
                        setattr(config, key, value)

    return resume_path, target_exp_dir

def get_loaders(config: Config, tokenizer: CipherTokenizer) -> tuple[DataLoader, DataLoader]:
    """Initialize training and validation data loaders with format detection."""
    train_path = Path(config.train_data_dir).resolve()
    valid_path = Path(config.valid_data_dir).resolve()

    is_arrow = (train_path / "dataset_info.json").exists() or any(train_path.glob("*.arrow"))

    if is_arrow:
        logger.info("Arrow format detected. Using PretokenizedCipherDataset.")
        train_ds = PretokenizedCipherDataset(train_path, max_seq_len=config.max_len, config=config)
        val_ds = PretokenizedCipherDataset(valid_path, max_seq_len=config.max_len, config=config)
    else:
        logger.info("Legacy format detected. Scanning directory for JSON/ZIPs...")
        train_files = DataManager.scan_directory(train_path)
        valid_files = DataManager.scan_directory(valid_path)

        train_ds = CipherDataset(train_files, max_seq_len=config.max_len, tokenizer=tokenizer, mode="train")
        val_ds = CipherDataset(valid_files, max_seq_len=config.max_len, tokenizer=tokenizer, mode="train")

    collate_fn = partial(
        DataManager.safe_pad_collate,
        pad_token_id=tokenizer.pad_token_id,
        ignore_index=-100,
    )

    num_workers = max(1, (os.cpu_count() or 1) - 4)
    loader_args = {
        "batch_size": config.batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": True if num_workers > 0 else False,
        "collate_fn": collate_fn,
    }

    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

    return train_loader, val_loader

def train_model(resume_arg: str | None = None, use_spaces: bool = False, device: str = "cuda") -> None:
    """Execute the full training pipeline."""
    config = Config()

    if use_spaces:
        logger.info("Mode: Training WITH spaces.")
        config.train_data_dir = config.tok_train_spaced
        config.valid_data_dir = config.tok_valid_spaced
        config.test_data_dir = config.tok_test_spaced
    else:
        logger.info("Mode: Training WITHOUT spaces (Normal).")
        config.train_data_dir = config.tok_train_normal
        config.valid_data_dir = config.tok_valid_normal
        config.test_data_dir = config.tok_test_normal

    run_type = "spaced" if use_spaces else "normal"
    resume_path, target_exp_dir = resolve_config(resume_arg, config, run_type)

    tokenizer = CipherTokenizer(config)
    train_loader, val_loader = get_loaders(config, tokenizer)

    model = MambaModel(
        vocab_size=tokenizer.vocab_size,
        char_offset=tokenizer.char_offset,
        config=config,
    ).to(device)

    if hasattr(torch, "compile"):
        logger.info("Compiling model with dynamic shapes for L4 performance...")
        model = torch.compile(
            model, 
            mode="reduce-overhead", 
            dynamic=True
        )

    trainer = MambaTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        save_path=Path(config.save_path),
        run_type=run_type,
        exp_dir=target_exp_dir,
        device=device,
    )

    if resume_path:
        trainer.load_checkpoint(resume_path)

    trainer.train(epochs=config.epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", nargs="?", const="auto", default=None)
    parser.add_argument("--spaces", action="store_true", help="Train on the dataset containing spaces.")
    args = parser.parse_args()

    train_model(resume_arg=args.resume, use_spaces=args.spaces)

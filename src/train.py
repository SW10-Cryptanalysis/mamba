import json
from torch.utils.data import DataLoader
import os
import argparse
from src.models.mamba import MambaModel
from src.utils.logging import get_logger
from pathlib import Path
from src.config import Config
from src.utils.data_manager import DataManager
from src.data.dataset import CipherDataset
from src.data.tokenizer import CipherTokenizer
from src.engine.trainer import MambaTrainer


logger = get_logger("train.py")

def resolve_config(
    resume_arg: str | None,
    config: Config,
) -> tuple[Path | None, Path | None]:
    """Handle checkpoint auto-detection and sync config from existing experiments."""
    save_path = Path(config.save_path)
    resume_path = None
    target_exp_dir = None

    if resume_arg == "auto":
        resume_path = DataManager.get_latest_checkpoint(save_path)
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

def get_loaders(
    config: Config,
    tokenizer: CipherTokenizer,
) -> tuple[DataLoader, DataLoader]:
    """Initialize training and validation data loaders."""
    train_files = DataManager.scan_directory(os.path.abspath(config.train_data_dir))
    valid_files = DataManager.scan_directory(os.path.abspath(config.valid_data_dir))

    if not (
        isinstance(config.unique_homophones, int) and isinstance(config.max_len, int)
    ):
        max_len, _ = DataManager.get_max_stats(train_files)
    else:
        max_len = config.max_len

    num_workers = max(1, (os.cpu_count() or 1) - 4)

    loader_args = {
        "batch_size": config.batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": True,
    }

    train_loader = DataLoader(
        CipherDataset(train_files, max_seq_len=max_len, tokenizer=tokenizer),
        shuffle=True,
        **loader_args,
    )
    val_loader = DataLoader(
        CipherDataset(valid_files, max_seq_len=max_len, tokenizer=tokenizer),
        shuffle=False,
        **loader_args,
    )
    return train_loader, val_loader

def train_model(resume_arg: str | None = None, device: str = "cuda") -> None:
    """Execute the full training pipeline."""
    config = Config()
    resume_path, target_exp_dir = resolve_config(resume_arg, config)

    tokenizer = CipherTokenizer(config)
    train_loader, val_loader = get_loaders(config, tokenizer)

    model = MambaModel(
        vocab_size=tokenizer.vocab_size,
        char_offset=tokenizer.char_offset,
        config=config,
    ).to(device)

    trainer = MambaTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        save_path=Path(config.save_path),
        exp_dir=target_exp_dir,
    )

    if resume_path:
        trainer.load_checkpoint(resume_path)

    trainer.train(epochs=config.epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", nargs="?", const="auto", default=None)
    args = parser.parse_args()

    train_model(resume_arg=args.resume)

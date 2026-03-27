import json
from pathlib import Path
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from src.models.mamba import get_model
from src.data.dataset import CipherPlainData
from src.data.pad_collator import PadCollator
from src.utils.logging import get_logger

logger = get_logger(__name__)

class MambaTrainer:
    def __init__(self, config, resume):
        self.cfg = config

        if resume:
            self.save_path = self._resolve_resume_path(resume)
            self.resume = True

            last_ckpt = get_last_checkpoint(str(self.save_path))
            if last_ckpt:
                logger.info(f"Resuming experiment: {self.save_path.name}")
                self.cfg = self._load_config(self.cfg, last_ckpt)
        else:
            self.save_path = Path(config.save_path)
            self.resume = False

        self.save_path.mkdir(parents=True, exist_ok=True)

        self.model = get_model(config)
        self.collator = PadCollator(pad_token_id=config.pad_token_id)
        self.train_ds = CipherPlainData(config, split="Training")
        self.eval_ds = CipherPlainData(config, split="Validation")
        self.trainer = self._setup_trainer()

    def _setup_trainer(self) -> Trainer:
        """Configures the HF Trainer."""
        args = TrainingArguments(
            output_dir=str(self.save_path),
            num_train_epochs=self.cfg.epochs,
            per_device_train_batch_size=self.cfg.batch_size,
            gradient_accumulation_steps=self.cfg.grad_accum,
            learning_rate=self.cfg.learning_rate,

            # Optimization & Precision
            bf16=True,
            tf32=True,
            optim="adamw_torch_fused",
            gradient_checkpointing=False,

            # Eval & Logging
            eval_strategy="steps",
            eval_steps=self.cfg.save_step,
            logging_steps=10, # Hardcoded or from config
            save_steps=self.cfg.save_step,

            # Checkpointing
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,

            dataloader_num_workers=8,
            dataloader_pin_memory=True,
        )

        return Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_ds,
            eval_dataset=self.eval_ds,
            data_collator=self.collator,
        )

    def _load_config(self, current_config, checkpoint_path):
        """Overwrites current_config with values found in the checkpoint."""
        config_file = Path(checkpoint_path) / "project_config.json"
        if config_file.exists():
            with open(config_file) as f:
                saved_data = json.load(f)

            for key, value in saved_data.items():
                if hasattr(current_config, key):
                    setattr(current_config, key, value)
            logger.info("Config successfully synchronized with checkpoint state.")
        return current_config

    def _save_config(self, save_path: Path):
        """Serializes the Config object to JSON."""
        config_dict = {k: v for k, v in vars(self.cfg).items() if not k.startswith("__")}
        with open(save_path / "project_config.json", "w") as f:
            json.dump(config_dict, f, indent=4, default=str)

    def _resolve_resume_path(self, resume_arg) -> Path:
        if isinstance(resume_arg, str):
            target = Path(resume_arg)
            if not target.exists():
                raise FileNotFoundError(f"Specified resume path {target} does not exist.")
            return target

        base_dir = self.cfg.outputs_dir / ("spaces" if self.cfg.use_spaces else "normal")

        if not base_dir.exists():
            raise FileNotFoundError(f"No previous runs found in {base_dir}")

        subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
        if not subdirs:
            raise FileNotFoundError(f"No run folders found in {base_dir}")

        latest_run = max(subdirs, key=lambda d: d.stat().st_mtime)
        logger.info(f"Auto-detected latest run: {latest_run.name}")
        return latest_run

    def run(self):
        """Execute training loop."""
        last_checkpoint = None

        if self.resume:
            last_checkpoint = get_last_checkpoint(str(self.save_path))
            if last_checkpoint:
                logger.info(f"Resuming training from {last_checkpoint}.")
            else:
                logger.error("Resume requested but no checkpoint found.")
                return None
        else:
            logger.info("Starting new training run.")
            self._save_config(self.save_path)

        self.trainer.train(resume_from_checkpoint=last_checkpoint)

        final_path = self.save_path / "final_model"
        self.trainer.save_model(final_path)

        self._save_config(final_path)

        logger.info(f"Model and project config saved to {final_path}")

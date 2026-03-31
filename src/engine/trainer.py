import json
from pathlib import Path
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from src.models.mamba import get_model
from src.data.dataset import CipherPlainData
from src.data.pad_collator import PadCollator
from src.config import Config
from src.utils.logging import get_logger

logger = get_logger(__name__)

class MambaTrainer:
    """Orchestrates the training pipeline for the Mamba2 cipher model.

    This class handles environment setup, dataset loading, configuration
    synchronization with checkpoints, and the execution of the Hugging Face
    Trainer loop.

    Attributes:
        cfg: Configuration object containing training hyperparameters.
        resume (bool): Whether the current run is resuming from a checkpoint.
        save_path (Path): Directory where checkpoints and logs are stored.
        model (Mamba2ForCausalLM): The Mamba2 model instance.
        collator (PadCollator): Data collator for dynamic padding.
        train_ds (CipherPlainData): Training dataset split.
        eval_ds (CipherPlainData): Validation dataset split.
        trainer (Trainer): The initialized Hugging Face Trainer instance.

    """

    def __init__(self, config: Config, resume: bool | str) -> None:
        """Initialize the trainer with config and sets up save/resume paths.

        Args:
            config: Configuration object containing paths and hyperparameters.
            resume: If True, auto-detects the latest run.
                If a string, uses that specific directory path.

        """
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

        self._inject_mamba2_kernels()
        self.model = get_model(config)
        self.collator = PadCollator(pad_token_id=config.pad_token_id)
        self.train_ds = CipherPlainData(config, split="Training")
        self.eval_ds = CipherPlainData(config, split="Validation")
        self.trainer = self._setup_trainer()

    def _inject_mamba2_kernels(self) -> None:
        """Force-injects Mamba2 CUDA kernels into the transformers modeling namespace."""
        try:
            import transformers.models.mamba2.modeling_mamba2 as mamba2_mod
            from mamba_ssm.ops.triton.selective_state_update import selective_state_update
            from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined
            from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

            mamba2_mod.selective_state_update = selective_state_update
            mamba2_mod.mamba_chunk_scan_combined = mamba_chunk_scan_combined
            mamba2_mod.mamba_split_conv1d_scan_combined = mamba_split_conv1d_scan_combined
            mamba2_mod.causal_conv1d_fn = causal_conv1d_fn
            mamba2_mod.causal_conv1d_update = causal_conv1d_update

            mamba2_mod.is_fast_path_available = True

            logger.info("Mamba2 Kernel Injection Successful: Fast Path Forced.")
        except Exception as e:
            logger.error(f"Mamba2 Kernel Injection failed: {e}")

    def _setup_trainer(self) -> Trainer:
        """Configure the Hugging Face TrainingArguments and Trainer.

        Returns:
            Trainer: A fully configured Hugging Face Trainer.

        """
        args = TrainingArguments(
            output_dir=str(self.save_path),
            num_train_epochs=self.cfg.scheduler_config.epochs,
            per_device_train_batch_size=self.cfg.scheduler_config.batch_size,
            gradient_accumulation_steps=self.cfg.scheduler_config.grad_accum,
            learning_rate=self.cfg.scheduler_config.learning_rate,

            # Optimization & Precision
            bf16=True,
            tf32=True,
            fsdp="full_shard auto_wrap",
            fsdp_config={
                "transformer_layer_cls_to_wrap": ["Mamba2Block"],
                "activation_checkpointing": True,
            },
            optim="adamw_torch_fused",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},


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

    def _load_config(self, current_config: Config, checkpoint_path: str) -> Config:
        """Overwrite current_config with values found in a checkpoint directory.

        Args:
            current_config: The current in-memory config object.
            checkpoint_path (Union[str, Path]): Path to the checkpoint folder
                containing 'project_config.json'.

        Returns:
            Any: The synchronized configuration object.

        """
        config_file = Path(checkpoint_path) / "project_config.json"
        if config_file.exists():
            with open(config_file) as f:
                saved_data = json.load(f)

            for key, value in saved_data.items():
                if hasattr(current_config, key):
                    setattr(current_config, key, value)
            logger.info("Config successfully synchronized with checkpoint state.")
        return current_config

    def _save_config(self, save_path: Path) -> None:
        """Serialize the current Config object to a JSON file.

        Args:
            save_path (Path): Directory where the config JSON will be saved.

        """
        config_dict = {k: v for k, v in vars(self.cfg).items()
                       if not k.startswith("__")}
        with open(save_path / "project_config.json", "w") as f:
            json.dump(config_dict, f, indent=4, default=str)

    def _resolve_resume_path(self, resume: bool | str) -> Path:
        """Determine the directory path for resuming a previous run.

        Args:
            resume: Either a specific path string or a
                boolean indicating auto-detection is required.

        Returns:
            Path: The resolved directory path.

        Raises:
            FileNotFoundError: If the specified path or auto-detected base
                directory does not exist.

        """
        if isinstance(resume, str):
            target = Path(resume)
            if not target.exists():
                raise FileNotFoundError(f"Specified resume path {target}"
                                        "does not exist.")
            return target

        base_dir = self.cfg.outputs_dir / (
            "spaces" if self.cfg.use_spaces else "normal"
        )

        if not base_dir.exists():
            raise FileNotFoundError(f"No previous runs found in {base_dir}")

        subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
        if not subdirs:
            raise FileNotFoundError(f"No run folders found in {base_dir}")

        latest_run = max(subdirs, key=lambda d: d.stat().st_mtime)
        logger.info(f"Auto-detected latest run: {latest_run.name}")
        return latest_run

    def run(self) -> None:
        """Execute the training loop.

        Handles the initial config save (for new runs), resumes from checkpoints
        if applicable, and saves the final model and config upon completion.
        """
        import transformers.models.mamba2.modeling_mamba2 as mamba2_mod
        print(f"--- FINAL VERIFICATION: Fast Path is {mamba2_mod.is_fast_path_available} ---")

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

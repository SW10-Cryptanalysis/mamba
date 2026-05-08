import os
import argparse
from src.config import Config
from src.engine.trainer import MambaTrainer

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--spaces", action="store_true")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        help="Resume latest or specify a folder",
    )
    parser.add_argument(
        "--task",
        choices=["causal", "mapping"],
        default="causal",
        help="The task to train on",
    )
    parser.add_argument("--truncated", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Begin the training process."""
    cmd_args = parse_args()

    config = Config(
        use_spaces=cmd_args.spaces,
        task=cmd_args.task,
        truncated=cmd_args.truncated,
    )
    config.load_homophones()

    trainer = MambaTrainer(config, resume=cmd_args.resume)
    trainer.run()


if __name__ == "__main__":
    main()

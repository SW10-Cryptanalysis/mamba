import os
import argparse
from src.config import Config
from src.engine.trainer import MambaTrainer

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

def main() -> None:
    """Begin the training process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--spaces", action="store_true")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        help="Resume latest or specify a folder",
    )
    cmd_args = parser.parse_args()

    config = Config()
    config.use_spaces = cmd_args.spaces

    trainer = MambaTrainer(config, resume=cmd_args.resume)
    trainer.run()

if __name__ == "__main__":
    main()

import argparse
from datasets import load_from_disk
from src.config import Config
from src.engine.solver import MambaCipherSolver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--spaces", action="store_true")
    args = parser.parse_args()

    config = Config()
    config.use_spaces = args.spaces
    config.load_homophones()

    solver = MambaCipherSolver(args.model_path, config)

    test_ds = load_from_disk(config.tokenized_dir / "Test")

    solver.evaluate(test_ds)

if __name__ == "__main__":
    main()

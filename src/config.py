from dataclasses import dataclass
import os
import json

@dataclass
class Config:
    d_model: int = 128
    n_layers: int = 4
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    
    batch_size: int = 16
    learning_rate: float = 1e-4
    epochs: int = 10
    num_workers: int = 4

    save_path: str = "./outputs"
    data_dir: str = "../Ciphers/"
    train_data_dir: str = os.path.join(data_dir, "Training")
    valid_data_dir: str = os.path.join(data_dir, "Validation")
    eval_data_dir: str = os.path.join(data_dir, "Test")
    homophone_file: str = "metadata.json"

    plain_vocab_size: int = 26
    unique_homophones: int = 500
    max_len: int = 1_000
    vocab_size: int = 0
    buffer: int = 1

    def __post_init__(self):
        homophone_path = os.path.join(self.data_dir, self.homophone_file)
        if os.path.exists(homophone_path):
            try:
                with open(homophone_path, "r") as f:
                    data = json.load(f)
                    self.unique_homophones = int(data.get("max_symbol_id", self.unique_homophones))
            except (json.JSONDecodeError, ValueError, IOError) as e:
                print(f"Warning - Could not parse {self.homophone_file}. Using default.")
                print(f"Error: {e}")
        self.vocab_size = self.unique_homophones + self.plain_vocab_size + self.buffer
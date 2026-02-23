from dataclasses import dataclass
import os

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
    train_data_dir: str = "./train_data"
    eval_data_dir: str = "./eval_data"
    homophone_file: str = "h_count"

    plain_vocab_size: int = 26
    unique_homophones: int = 500
    max_len: int = 1_000
    vocab_size: int = 0
    buffer: int = 1

    def __post_init__(self):
        homophone_path = os.path.join(self.train_data_dir, self.homophone_file)
        if os.path.exists(homophone_path):
            try:
                with open(homophone_path, "r") as f:
                    self.unique_homophones = int(f.read().strip())
            except (ValueError, IOError) as e:
                print(f"Warning - Could not read file: {self.homophone_file}")
                print(f"Using default value: {self.unique_homophones}")
                print(f"{e}")

        self.vocab_size = self.unique_homophones + self.plain_vocab_size + self.buffer
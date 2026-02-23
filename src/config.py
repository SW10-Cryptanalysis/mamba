from dataclasses import dataclass

@dataclass
class Config:
    plain_vocab_size: int = 26
    unique_homophones: int = 500
    max_len: int = 1_000

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
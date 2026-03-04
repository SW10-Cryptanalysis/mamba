from torch.utils.data import Dataset
from ..utils.data_manager import DataManager
from src.utils.logging import get_logger
logger = get_logger("data/dataset.py")

class CipherDataset(Dataset):
    """
    A unified Dataset for Cipher tasks.
    
    Modes:
        'train': Returns (cipher_tensor, plain_tensor)
        'eval':  Returns (cipher_tensor, original_plaintext, metadata)
    """
    def __init__(self, file_paths, max_seq_len, tokenizer, mode="train"):
        self.file_paths = file_paths
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.mode = mode

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path, internal_name = self.file_paths[idx]
        
        try:
            data = DataManager.load_sample(path, internal_name)
            
            ciphertext = data["ciphertext"]
            if isinstance(ciphertext, str):
                ciphertext = [int(x) for x in ciphertext.split()]
            
            cipher_tensor = self.tokenizer.pad_sequence(ciphertext, self.max_seq_len)

            if self.mode == "eval":
                metadata = {
                    "path": str(path), 
                    "internal_name": internal_name if internal_name is not None else ""
                }
                return cipher_tensor, data["plaintext"], metadata
            
            encoded_plain = self.tokenizer.encode(data["plaintext"])
            plain_tensor = self.tokenizer.pad_sequence(encoded_plain, self.max_seq_len)
            
            return cipher_tensor, plain_tensor

        except Exception as e:
            logger.error(f"Error loading index {idx}: {e}")
            raise e
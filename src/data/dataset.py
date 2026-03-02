import torch
from torch.utils.data import Dataset
from ..utils.data_manager import DatasetManager
import logging

logger = logging.getLogger("data/dataset.py")

class CipherDataset(Dataset):
    """
    A unified Dataset for Cipher tasks.
    
    Modes:
        'train': Returns (cipher_tensor, plain_tensor)
        'eval':  Returns (cipher_tensor, original_plaintext, metadata)
    """
    def __init__(self, file_paths, max_seq_len, config, mode="train"):
        self.file_paths = file_paths
        self.max_seq_len = max_seq_len
        self.mode = mode
        
        self.mapping = {chr(i + 97): i for i in range(26)}
        
        self.sep_token = config.unique_homophones + 1
        self.char_offset = self.sep_token + 1

    def __len__(self):
        return len(self.file_paths)

    def _pad_trunc(self, list_data):
        """Standardizes sequence length."""
        if len(list_data) > self.max_seq_len:
            return list_data[:self.max_seq_len]
        return list_data + [0] * (self.max_seq_len - len(list_data))

    def __getitem__(self, idx):
        path, internal_name = self.file_paths[idx]
        
        try:
            data = DatasetManager.load_sample(path, internal_name)
            
            ciphertext = data["ciphertext"]
            if isinstance(ciphertext, str):
                ciphertext = [int(x) for x in ciphertext.split()]
            cipher_tensor = torch.tensor(self._pad_trunc(ciphertext), dtype=torch.long)

            if self.mode == "eval":
                metadata = {"path": path, "internal_name": internal_name if internal_name is not None else ""}
                return cipher_tensor, data["plaintext"], metadata
            
            plaintext = data["plaintext"]
            encoded_plain = [
                self.mapping[c] + self.char_offset 
                for c in plaintext.lower() 
                if c in self.mapping
            ]
            plain_tensor = torch.tensor(self._pad_trunc(encoded_plain), dtype=torch.long)
            
            return cipher_tensor, plain_tensor

        except Exception as e:
            logger.error(f"Error loading index {idx}: {e}")
            raise e
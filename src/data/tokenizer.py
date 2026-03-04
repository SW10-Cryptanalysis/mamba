import torch

class CipherTokenizer:
    def __init__(self, config):
        self.config = config
        self.pad_token_id = 0
        self.sep_token = config.unique_homophones + 1
        self.char_offset = self.sep_token + 1

        self.char_to_id = {chr(i + 97): i for i in range(26)}
        self.id_to_char = {i + self.char_offset: chr(i + 97) for i in range(26)}

    def pad_sequence(self, ids: list[int], max_len: int) -> torch.Tensor:
        """Handles Truncation and Padding, returning a LongTensor."""
        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids = ids + [self.pad_token_id] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def encode(self, text: str) -> list[int]:
        """Convert 'abc' -> [offset, offset+1, offset+2]"""
        return [
            self.char_to_id[c] + self.char_offset
            for c in text.lower()
            if c in self.char_to_id
        ]

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        """Convert IDs back to string, ignoring PAD and SEP."""
        if isinstance(ids, torch.Tensor):
            ids = ids.view(-1).tolist()
        return "".join([self.id_to_char[i] for i in ids if i in self.id_to_char])

    @property
    def vocab_size(self) -> int:
        return self.char_offset + self.config.plain_vocab_size + self.config.buffer

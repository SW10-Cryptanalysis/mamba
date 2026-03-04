import torch
from src.config import Config

class CipherTokenizer:
    """Tokenizer for converting between cipher homophones and plaintext characters.

    Attributes:
        config (Config): Configuration object containing vocabulary sizes and offsets.
        pad_token_id (int): ID used for padding sequences (default is 0).
        sep_token (int): ID used as a separator, calculated based on cipher vocabulary.
        char_offset (int): The starting index for plaintext character IDs to avoid
            collisions with cipher homophones.
        char_to_id (dict[str, int]): Mapping from lowercase characters to raw indices.
        id_to_char (dict[int, str]): Mapping from offset integer IDs back to
            plaintext characters.

    """

    def __init__(self, config: Config) -> None:
        """Initialize tokenizer with vocabulary offsets.

        Args:
            config: Configuration object containing vocabulary sizes and offsets.

        """
        self.config = config
        self.pad_token_id = 0
        self.sep_token = config.unique_homophones + 1
        self.char_offset = self.sep_token + 1

        self.char_to_id = {chr(i + 97): i for i in range(26)}
        self.id_to_char = {i + self.char_offset: chr(i + 97) for i in range(26)}

    def pad_sequence(self, ids: list[int], max_len: int) -> torch.Tensor:
        """Handle truncation and padding, returning a LongTensor.

        Args:
            ids: List of integer token IDs.
            max_len: Desired sequence length.

        Returns:
            A padded or truncated torch.Tensor of dtype long.

        """
        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids = ids + [self.pad_token_id] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def encode(self, text: str) -> list[int]:
        """Convert plaintext string to a list of integer IDs.

        Args:
            text: The plaintext string to encode.

        Returns:
            List of integer IDs shifted by the character offset.

        """
        return [
            self.char_to_id[c] + self.char_offset
            for c in text.lower()
            if c in self.char_to_id
        ]

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        """Convert IDs back to string, ignoring PAD and SEP tokens.

        Args:
            ids: List of IDs or a torch.Tensor to decode.

        Returns:
            The decoded plaintext string.

        """
        if isinstance(ids, torch.Tensor):
            ids = ids.view(-1).tolist()
        return "".join([self.id_to_char[i] for i in ids if i in self.id_to_char])

    @property
    def vocab_size(self) -> int:
        """Calculate the total vocabulary size including offsets and buffer.

        Returns:
            The total integer size of the vocabulary.

        """
        return self.char_offset + self.config.plain_vocab_size + self.config.buffer

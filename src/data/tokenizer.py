import torch
from src.config import Config

class CipherTokenizer:
    """Tokenizer for converting between cipher homophones and plaintext characters.

    Attributes:
        config (Config): Configuration object containing vocabulary sizes and offsets.
        pad_token_id (int): ID used for padding sequences (0).
        sep_token_id (int): ID used to separate cipher from plain text.
        space_token_id (int): ID specifically for the space character.
        eos_token_id (int): ID used to signify the end of a sequence.
        char_offset (int): The starting index for plaintext character IDs to avoid
            collisions with cipher homophones.
        char_to_id (dict[str, int]): Mapping from characters to their integer IDs.
        id_to_char (dict[int, str]): Mapping from integer IDs back to characters.

    """

    def __init__(self, config: Config) -> None:
        """Initialize the tokenizer with offsets and character mappings.

        Args:
            config: Configuration object used to determine token offsets based
                on the number of unique homophones in the cipher.

        """
        self.config = config
        self.pad_token_id = 0

        self.sep_token_id = config.unique_homophones + 1
        self.space_token_id = config.unique_homophones + 2
        self.eos_token_id = config.unique_homophones + 4
        self.char_offset = config.unique_homophones + 5

        self.char_to_id = {" ": self.space_token_id}
        self.id_to_char = {self.space_token_id: " "}

        for i in range(26):
            char = chr(ord("a") + i)
            token_id = self.char_offset + i
            self.char_to_id[char] = token_id
            self.id_to_char[token_id] = char

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        """Convert token IDs back to a string, filtering out special control tokens.

        Args:
            ids: A list of integer IDs or a torch.Tensor to be decoded.
                Handles both 1D and multi-dimensional tensors by flattening.

        Returns:
            str: The decoded string containing only mapped characters,
                excluding PAD, SEP, and EOS tokens.

        """
        if isinstance(ids, torch.Tensor):
            ids = ids.view(-1).tolist()

        special_tokens = {
            self.pad_token_id,
            self.sep_token_id,
            self.eos_token_id,
        }

        return "".join([
            self.id_to_char[i]
            for i in ids
            if i in self.id_to_char and i not in special_tokens
        ])

    @property
    def vocab_size(self) -> int:
        """Calculate the total vocabulary size including offsets and buffer.

        Returns:
            The total integer size of the vocabulary.

        """
        return self.char_offset + self.config.plain_vocab_size + self.config.buffer

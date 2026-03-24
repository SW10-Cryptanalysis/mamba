import pytest
from src.data.tokenizer import CipherTokenizer
from src.config import Config

@pytest.fixture
def tokenizer():
    """Create a tokenizer with a controlled config for testing."""
    config = Config()
    config.unique_homophones = 100
    config.plain_vocab_size = 26
    config.buffer = 10
    return CipherTokenizer(config)

def test_initialization_offsets(tokenizer):
    """Verify that tokens don't overlap using dynamic attributes."""
    assert tokenizer.pad_token_id == 0
    # SEP is homophones + 1
    assert tokenizer.sep_token_id == tokenizer.config.unique_homophones + 1
    # Space is SEP + 1
    assert tokenizer.space_token_id == tokenizer.config.unique_homophones + 2
    # Alphabet starts after Space
    assert tokenizer.char_offset == tokenizer.config.unique_homophones + 5

    assert tokenizer.id_to_char[tokenizer.char_offset] == "a"
    assert tokenizer.id_to_char[tokenizer.space_token_id] == " "

def test_decode_filtering(tokenizer):
    """Verify decoding ignores PAD and SEP but KEEPS spaces."""
    ids = [
        tokenizer.char_offset,      # 'a'
        tokenizer.char_offset + 1,  # 'b'
        tokenizer.sep_token_id,     # SEP (filter out)
        0                           # PAD (filter out)
    ]
    decoded = tokenizer.decode(ids)
    assert decoded == "ab"

def test_vocab_size_calculation(tokenizer):
    """Verify the property returns the correct total size dynamically."""
    # current logic: char_offset (103) + plain_vocab (26) + buffer (10) = 139
    expected = tokenizer.char_offset + tokenizer.config.plain_vocab_size + tokenizer.config.buffer
    assert tokenizer.vocab_size == expected

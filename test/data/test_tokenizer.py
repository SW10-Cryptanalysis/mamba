import pytest
import torch
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
    """Verify that tokens don't overlap."""
    assert tokenizer.pad_token_id == 0
    assert tokenizer.sep_token_id == 101
    assert tokenizer.char_offset == 102
    # First char 'a' should be at 102
    assert 102 in tokenizer.id_to_char
    assert tokenizer.id_to_char[102] == "a"

def test_encode_shifting(tokenizer):
    """Verify that plaintext is shifted correctly by the offset."""
    text = "abc"
    encoded = tokenizer.encode(text)

    # 'a' is index 0 in char_to_id. 0 + 102 = 102.
    expected = [102, 103, 104]
    assert encoded == expected

def test_decode_filtering(tokenizer):
    """Verify decoding ignores PAD (0) and SEP (101) tokens."""
    # [102('a'), 103('b'), 101(SEP), 0(PAD)]
    ids = [102, 103, 101, 0]
    decoded = tokenizer.decode(ids)
    assert decoded == "ab"

def test_round_trip(tokenizer):
    """Ensure encoding then decoding returns the original string (lowercase)."""
    original = "HelloMamba"
    encoded = tokenizer.encode(original)
    decoded = tokenizer.decode(encoded)
    assert decoded == original.lower()

def test_pad_sequence_truncation(tokenizer):
    """Verify that sequences longer than max_len are truncated."""
    ids = [1, 2, 3, 4, 5]
    tensor = tokenizer.pad_sequence(ids, max_len=3)
    assert tensor.shape[0] == 3
    assert tensor.tolist() == [1, 2, 3]

def test_pad_sequence_padding(tokenizer):
    """Verify that sequences shorter than max_len are padded with pad_token_id."""
    ids = [1, 2]
    tensor = tokenizer.pad_sequence(ids, max_len=4)
    assert tensor.tolist() == [1, 2, 0, 0]
    assert tensor.dtype == torch.long

def test_vocab_size_calculation(tokenizer):
    """Verify the property returns the correct total size."""
    # offset(102) + plain_vocab(26) + buffer(10) = 138
    assert tokenizer.vocab_size == 138

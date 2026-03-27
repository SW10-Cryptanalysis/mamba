import pytest
import torch
from src.data.pad_collator import PadCollator

@pytest.fixture
def collator():
    # Standard collator with pad=0 and no truncation
    return PadCollator(pad_token_id=0)

@pytest.fixture
def mock_features():
    return [
        {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},        # Length 3
        {"input_ids": [4, 5, 6, 7, 8], "labels": [4, 5, 6, 7, 8]},  # Length 5
    ]

def test_basic_padding(collator, mock_features):
    """Verify that sequences are padded to the same length (multiple of 8)."""
    output = collator(mock_features)

    # Longest seq is 5. Next multiple of 8 is 8.
    assert output["input_ids"].shape == (2, 8)
    assert output["labels"].shape == (2, 8)

    # Check if first sequence (len 3) was padded with 5 zeros
    assert torch.all(output["input_ids"][0, 3:] == 0)
    # Check if labels were padded with -100 (ignore_index)
    assert torch.all(output["labels"][0, 3:] == -100)

def test_multiple_of_eight_logic():
    """Verify that if sequence is exactly 8, it doesn't add 8 more."""
    collator = PadCollator(pad_token_id=0)
    features = [{"input_ids": [1]*8, "labels": [1]*8}]

    output = collator(features)
    assert output["input_ids"].shape[1] == 8

def test_truncation():
    """Verify that max_context limits the sequence length."""
    collator = PadCollator(pad_token_id=0, max_context=4)
    features = [{"input_ids": [1, 2, 3, 4, 5, 6], "labels": [1, 2, 3, 4, 5, 6]}]

    output = collator(features)
    # Truncated to 4, then padded to 8 (multiple of 8)
    assert output["input_ids"].shape[1] == 8
    # Ensure the 5th and 6th elements were dropped before padding
    assert output["input_ids"][0, 3] == 4
    assert output["input_ids"][0, 4] == 0

def test_attention_mask(collator, mock_features):
    """Verify mask is 1 for data and 0 for padding."""
    output = collator(mock_features)
    mask = output["attention_mask"]

    # First sample was length 3. Mask should be [1, 1, 1, 0, 0, 0, 0, 0]
    expected_mask = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0])
    assert torch.equal(mask[0], expected_mask)

def test_empty_features(collator):
    """Verify behavior with empty input."""
    output = collator([])
    assert output["input_ids"].shape == (0, 0)
    assert isinstance(output["input_ids"], torch.Tensor)

def test_ignore_index_consistency(collator):
    """Ensure labels use -100 for padding, even if pad_token_id is different."""
    custom_collator = PadCollator(pad_token_id=99)
    features = [{"input_ids": [1, 2], "labels": [1, 2]}]

    output = custom_collator(features)
    # Labels should still use -100 for padding at index 2 onwards
    assert output["labels"][0, 2] == -100

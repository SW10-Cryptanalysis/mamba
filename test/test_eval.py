from src.eval import decode_plain, ser

def test_decode_plain():
    """Verify that indices are converted to the correct letters with offset."""
    offset = 500
    indices = [500, 501, 502, 525]
    expected = "abcz"
    assert decode_plain(indices, offset) == expected

def test_decode_plain_unknown():
    """Verify that out-of-bounds indices return a question mark."""
    offset = 1000
    indices = [1000, 99]
    assert decode_plain(indices, offset) == "a?"

def test_ser():
    """If 2 out of 4 symbols are wrong, SER should be 0.5."""
    pred = [1, 9, 3, 9]
    target = [1, 2, 3, 4]
    assert ser(pred, target) == 0.5

def test_ser_empty():
    """Verify the edge case for empty input returns 0.0 as defined."""
    assert ser([], []) == 0.0

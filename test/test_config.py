import pytest
from dataclasses import dataclass
from typing import Any

from src.config import Config, MambaConfig, CosineSchedulerConfig


@dataclass
class SuccessTestCase:
    """Dataclass defining parameters for successful config initialization."""

    use_spaces: bool
    mock_json: str
    expected_suffix: str
    expected_unique: int
    expected_vocab_size: int
    expected_sep_token_id: int
    expected_eos_token_id: int
    expected_pad_token_id: int
    expected_max_len: int
    expected_save_mode: str


@dataclass
class FailureTestCase:
    """Dataclass defining parameters for failed config initialization."""

    name: str
    file_exception: Exception | None
    json_content: str


@pytest.mark.parametrize(
    "tc",
    [
        SuccessTestCase(
            use_spaces=False,
            mock_json='{"max_symbol_id": 100}',
            expected_suffix="normal",
            expected_unique=100,
            expected_sep_token_id=101,
            expected_eos_token_id=104,
            expected_pad_token_id=0,
            expected_vocab_size=141,
            expected_max_len=20139,
            expected_save_mode="normal",
        ),
        SuccessTestCase(
            use_spaces=True,
            mock_json='{"max_symbol_id": 2503}',
            expected_suffix="spaced",
            expected_unique=2503,
            expected_sep_token_id=2504,
            expected_eos_token_id=2507,
            expected_pad_token_id=0,
            expected_vocab_size=2544,
            expected_max_len=26167,
            expected_save_mode="spaces",
        ),
    ],
    ids=lambda tc: f"use_spaces_{tc.use_spaces}",
)
def test_config_initialization_success(tc: SuccessTestCase, mocker: Any) -> None:
    """Verifies successful loading of properties, dynamic vocabulary sizing, and paths."""
    mocker.patch("builtins.open", mocker.mock_open(read_data=tc.mock_json))

    cfg = Config(use_spaces=tc.use_spaces)

    assert tc.expected_suffix in str(cfg.tokenized_dir)
    assert cfg.unique_homophones == tc.expected_unique
    assert cfg.mamba_config.vocab_size == tc.expected_vocab_size
    assert cfg.mamba_config.sep_token_id == tc.expected_sep_token_id
    assert cfg.mamba_config.eos_token_id == tc.expected_eos_token_id
    assert cfg.mamba_config.pad_token_id == tc.expected_pad_token_id
    assert cfg.char_offset == tc.expected_eos_token_id + 1
    assert cfg.bos_token_id == cfg.space_token_id + 1
    assert cfg.max_len == tc.expected_max_len
    assert tc.expected_save_mode in str(cfg.save_path)


@pytest.mark.parametrize(
    "tc",
    [
        FailureTestCase(
            name="os_error_missing_file",
            file_exception=OSError("File missing"),
            json_content="",
        ),
        FailureTestCase(
            name="key_error_wrong_json",
            file_exception=None,
            json_content='{"wrong_key": 100}',
        ),
        FailureTestCase(
            name="value_error_invalid_json",
            file_exception=None,
            json_content="invalid json format",
        ),
    ],
    ids=lambda tc: tc.name,
)
def test_config_initialization_failures(tc: FailureTestCase, mocker: Any) -> None:
    """Verifies RuntimeError and logging upon missing or invalid homophone data."""
    mock_logger = mocker.patch("src.config.logger.error")
    mock_file = mocker.patch(
        "builtins.open", mocker.mock_open(read_data=tc.json_content)
    )

    if tc.file_exception:
        mock_file.side_effect = tc.file_exception

    with pytest.raises(RuntimeError) as exc_info:
        Config()

    assert "Aborting initialization" in str(exc_info.value)
    assert mock_logger.call_count == 1


def test_mamba_config_defaults() -> None:
    """Verifies default initialization fields of MambaConfig."""
    config = MambaConfig()

    assert config.num_heads == 16
    assert config.hidden_size == 1024
    assert config.expand == 1
    assert config.hidden_act == "silu"


def test_cosine_scheduler_config_defaults() -> None:
    """Verifies default initialization fields of CosineSchedulerConfig."""
    config = CosineSchedulerConfig()

    assert config.learning_rate == 5e-4
    assert config.batch_size == 4
    assert config.epochs == 5
    assert config.warmup_ratio == 0.1

import pytest
from dataclasses import dataclass

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


@dataclass
class FailureTestCase:
    """Dataclass defining parameters for failed config initialization."""

    file_exception: Exception | None
    json_content: str
    expected_log_call_count: int


@pytest.mark.parametrize(
    "tc",
    [
        SuccessTestCase(
            use_spaces=False,
            mock_json='{"max_symbol_id": 100}',
            expected_suffix="normal",
            expected_unique=100,
            expected_vocab_size=181,
            expected_sep_token_id=101,
            expected_eos_token_id=104,
            expected_pad_token_id=0,
        ),
        SuccessTestCase(
            use_spaces=True,
            mock_json='{"max_symbol_id": 2503}',
            expected_suffix="spaced",
            expected_unique=2503,
            expected_vocab_size=2584,
            expected_sep_token_id=2504,
            expected_eos_token_id=2507,
            expected_pad_token_id=0,
        ),
    ],
)
def test_config_initialization_success(tc: SuccessTestCase, mocker):
    """Verifies successful loading of properties and dynamic vocabulary sizing."""
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


@pytest.mark.parametrize(
    "tc",
    [
        FailureTestCase(
            file_exception=OSError("File missing"),
            json_content="",
            expected_log_call_count=1,
        ),
        FailureTestCase(
            file_exception=None,
            json_content='{"wrong_key": 100}',
            expected_log_call_count=1,
        ),
        FailureTestCase(
            file_exception=None,
            json_content="invalid json format",
            expected_log_call_count=1,
        ),
    ],
)
def test_config_initialization_failures(tc: FailureTestCase, mocker):
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
    assert mock_logger.call_count == tc.expected_log_call_count


def test_mamba_config_defaults():
    """Verifies default initialization fields of MambaConfig."""
    config = MambaConfig()

    assert config.num_heads == 128
    assert config.hidden_size == 4096
    assert config.expand == 2
    assert config.hidden_act == "silu"


def test_cosine_scheduler_config_defaults():
    """Verifies default initialization fields of CosineSchedulerConfig."""
    config = CosineSchedulerConfig()

    assert config.learning_rate == 5e-4
    assert config.batch_size == 128
    assert config.epochs == 30
    assert config.warmup_ratio == 0.1

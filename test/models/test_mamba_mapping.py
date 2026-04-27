import pytest
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

from src.models.mamba_mapping import Mamba2ForMapping, get_mapping_model

MODULE_PATH = "src.models.mamba_mapping"


@dataclass
class DummyMambaConfig:
    """Mock configuration to satisfy the asdict() call in the factory."""

    hidden_size: int = 16
    num_hidden_layers: int = 1
    vocab_size: int = 100
    n_groups: int = 1


@dataclass
class DummyOutput:
    """A real object to replace the globally mocked SequenceClassifierOutput."""

    loss: Any = None
    logits: Any = None
    hidden_states: Any = None


@pytest.fixture(autouse=True)
def unmock_sequence_output(mocker: Any) -> None:
    """
    Because the test environment globally mocks `transformers`,
    we must inject a real dataclass into the module's namespace
    so our tensors don't get swallowed into a MagicMock at the end.
    """
    mocker.patch(f"{MODULE_PATH}.SequenceClassifierOutput", DummyOutput)


@dataclass
class ForwardTestCase:
    """Dataclass for clean forward pass parameterization."""

    name: str
    input_ids: torch.Tensor
    labels: torch.Tensor | None
    expected_logits_size: int
    has_loss: bool


def get_forward_cases() -> list[ForwardTestCase]:
    return [
        ForwardTestCase(
            name="inference_no_labels",
            input_ids=torch.tensor([[1, 2, 0]]),
            labels=None,
            expected_logits_size=2,
            has_loss=False,
        ),
        ForwardTestCase(
            name="training_with_labels",
            input_ids=torch.tensor([[1, 2, 3]]),
            labels=torch.tensor([[10, 11, 12]]),
            expected_logits_size=3,
            has_loss=True,
        ),
        ForwardTestCase(
            name="multi_batch_and_duplicates",
            input_ids=torch.tensor([[1, 1, 2, 0], [3, 4, 0, 0]]),
            labels=torch.tensor([[10, 11, -100], [12, 13, -100]]),
            expected_logits_size=4,
            has_loss=True,
        ),
    ]


@pytest.fixture
def base_config() -> MagicMock:
    """
    Provides a completely isolated mock configuration.
    This bypasses any global transformers mocks in the environment.
    """
    config = MagicMock()
    config.hidden_size = 16
    config.num_hidden_layers = 1
    config.vocab_size = 100
    return config


class TestMamba2ForMapping:
    """Test suite for the Mamba2ForMapping architecture."""

    def test_initialization(self, base_config: MagicMock) -> None:
        """Verifies the classification head scales cleanly from the config."""
        model = Mamba2ForMapping(base_config, num_labels=26)

        assert model.num_labels == 26
        assert isinstance(model.classifier, nn.Linear)
        assert model.classifier.in_features == 16
        assert model.classifier.out_features == 26

    @pytest.mark.parametrize("case", get_forward_cases(), ids=lambda c: c.name)
    def test_forward_pass(
        self,
        case: ForwardTestCase,
        base_config: MagicMock,
    ) -> None:
        """Verifies tensor shapes, label alignment, and loss presence."""
        model = Mamba2ForMapping(base_config, num_labels=26)

        batch_size, seq_len = case.input_ids.shape
        real_hidden_states = torch.randn(batch_size, seq_len, 16)

        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = real_hidden_states
        mock_outputs.hidden_states = None
        model.mamba2 = MagicMock(return_value=mock_outputs)

        output = model(input_ids=case.input_ids, labels=case.labels)

        assert output.logits.shape == (case.expected_logits_size, 26)

        if case.has_loss:
            assert output.loss is not None
            assert isinstance(output.loss, torch.Tensor)
            assert output.loss.ndim == 0
        else:
            assert output.loss is None

    def test_pooling_mathematics(self, base_config: MagicMock) -> None:
        """Mathematically proves the hidden states are correctly mean-pooled."""
        model = Mamba2ForMapping(base_config, num_labels=26)

        input_ids = torch.tensor([[1, 1]])

        real_hidden_states = torch.ones(1, 2, 16)
        real_hidden_states[0, 1, :] = 3.0

        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = real_hidden_states
        model.mamba2 = MagicMock(return_value=mock_outputs)

        output = model(input_ids=input_ids)

        expected_pooled_tensor = torch.full((1, 16), 2.0)
        expected_logits = model.classifier(expected_pooled_tensor)

        torch.testing.assert_close(output.logits, expected_logits)


def test_get_mapping_model(mocker: Any) -> None:
    """Verifies the factory function applies correct types and logs parameters."""
    mock_logger = mocker.patch(f"{MODULE_PATH}.logger")

    class IsolatedConfig:
        def __init__(self, **kwargs: Any):
            self.__dict__.update(kwargs)

    mocker.patch(f"{MODULE_PATH}.Mamba2Config", IsolatedConfig)
    mocker.patch(f"{MODULE_PATH}.Mamba2Model")

    config = DummyMambaConfig(hidden_size=32)
    model = get_mapping_model(config)  # type: ignore

    assert model.config.hidden_size == 32
    assert model.config.torch_dtype == torch.bfloat16

    assert mock_logger.info.call_count == 2
    mock_logger.info.assert_any_call("Mamba2 Mapping Model loaded!")

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import numpy as np

import pytest

from src.engine.trainer import MambaTrainer


@dataclass
class SchedulerConfig:
    """Dummy scheduler configuration."""

    epochs: int = 1
    batch_size: int = 2
    grad_accum: int = 1
    learning_rate: float = 0.001
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.1


@dataclass
class MockConfig:
    """Dummy configuration structure mimicking the real Config."""

    task: Literal["causal", "mapping"]
    save_path: str
    outputs_dir: str
    use_spaces: bool
    mamba_config: dict
    pad_token_id: int
    save_step: int
    scheduler_config: SchedulerConfig
    tokenized_dir: Path
    max_len: int = 100
    sep_token_id: int = 1
    space_token_id: int = 2
    eos_token_id: int = 4
    bos_token_id: int = 3
    char_offset: int = 5

@pytest.fixture
def base_config(tmp_path: Path) -> MockConfig:
    """Provides a clean instance of MockConfig for each test."""
    return MockConfig(
        task="causal",
        save_path=str(tmp_path / "save"),
        outputs_dir=str(tmp_path / "outputs"),
        use_spaces=False,
        mamba_config={"d_model": 128},
        pad_token_id=0,
        save_step=10,
        scheduler_config=SchedulerConfig(),
        tokenized_dir=tmp_path / "tokenized",
    )


@dataclass
class MockEvalPrediction:
    """
    Mock prediction object.
    Matches the Hugging Face EvalPrediction where predictions/labels
    can be raw arrays or tuples of arrays.
    """

    predictions: np.ndarray | tuple[np.ndarray, ...]
    label_ids: np.ndarray | tuple[np.ndarray, ...]


@dataclass
class ComputeMappingMetricsTestCase:
    """Dataclass for testing compute_mapping_metrics."""

    name: str
    eval_pred: MockEvalPrediction
    expected: dict[str, float]


def create_mock_logits(classes: list[int]) -> np.ndarray:
    """Creates a (N, 26) logit matrix where the specified classes are winners."""
    logits = np.zeros((len(classes), 26))
    for i, cls in enumerate(classes):
        valid_cls = max(0, min(cls, 25))
        logits[i, valid_cls] = 1.0
    return logits


compute_mapping_metrics_cases = [
    ComputeMappingMetricsTestCase(
        name="perfect_prediction_standard_array",
        eval_pred=MockEvalPrediction(
            predictions=create_mock_logits([0, 1, 2]),
            label_ids=np.array([0, 1, 2]),
        ),
        expected={"accuracy": 1.0},
    ),
    ComputeMappingMetricsTestCase(
        name="tuple_predictions_handling",
        eval_pred=MockEvalPrediction(
            # Wrapped in tuple: mimics HF returning (logits, hidden_states)
            predictions=(create_mock_logits([0, 1, 2]),),
            label_ids=np.array([0, 1, 2]),
        ),
        expected={"accuracy": 1.0},
    ),
    ComputeMappingMetricsTestCase(
        name="both_are_tuples",
        eval_pred=MockEvalPrediction(
            predictions=(create_mock_logits([0, 5]),),
            label_ids=(np.array([0, 1]),),
        ),
        # Index 0 matches, Index 1 fails. 1/2 = 0.5
        expected={"accuracy": 0.5},
    ),
    ComputeMappingMetricsTestCase(
        name="with_ignored_indices_and_tuples",
        eval_pred=MockEvalPrediction(
            predictions=(create_mock_logits([0, 5, 10]),),
            label_ids=(np.array([0, 5, -100]),),
        ),
        # Two considered, both correct. 2/2 = 1.0
        expected={"accuracy": 1.0},
    ),
    ComputeMappingMetricsTestCase(
        name="only_ignored_indices",
        eval_pred=MockEvalPrediction(
            predictions=create_mock_logits([0, 5, 10]),
            label_ids=np.array([-100, -100, -100]),
        ),
        # All ignored. 0/3 = 0.0
        expected={"accuracy": 0.0},
    ),
]


@pytest.mark.parametrize("case", compute_mapping_metrics_cases, ids=lambda c: c.name)
def test_compute_mapping_metrics(
    case: ComputeMappingMetricsTestCase, mocker: Any, base_config: MockConfig
) -> None:
    """Tests the compute_mapping_metrics function."""
    mocker.patch("src.engine.trainer.get_mapping_model", return_value=mocker.Mock())
    base_config.task = "mapping"
    trainer_instance = MambaTrainer(base_config, resume=False)  # type: ignore
    assert trainer_instance.compute_metrics is not None
    result = trainer_instance.compute_metrics(case.eval_pred)  # type: ignore

    assert result == case.expected


@pytest.fixture(autouse=True)
def mock_dependencies(mocker: Any) -> dict[str, Any]:
    """Mocks class-level dependencies and strictly isolates hardware injections."""
    return {
        "model": mocker.patch(
            "src.engine.trainer.get_model", return_value=mocker.Mock()
        ),
        "collator": mocker.patch(
            "src.engine.trainer.PadCollator", return_value=mocker.Mock()
        ),
        "data": mocker.patch(
            "src.engine.trainer.CipherPlainData", return_value=mocker.Mock()
        ),
        "trainer": mocker.patch(
            "src.engine.trainer.Trainer", return_value=mocker.Mock()
        ),
        "args": mocker.patch(
            "src.engine.trainer.TrainingArguments", return_value=mocker.Mock()
        ),
        "ckpt": mocker.patch(
            "src.engine.trainer.get_last_checkpoint", return_value=None
        ),
        "inject": mocker.patch(
            "src.engine.trainer.MambaTrainer._inject_mamba2_kernels"
        ),
    }


@pytest.fixture
def trainer_instance(
    base_config: MockConfig, mock_dependencies: dict[str, Any], tmp_path: Path
) -> MambaTrainer:
    """Returns a generic initialized MambaTrainer object."""
    return MambaTrainer(base_config, resume=False)  # type: ignore


@dataclass
class InitTestCase:
    """Data structure for testing MambaTrainer initialization scenarios."""

    name: str
    resume: bool | str
    has_checkpoint: bool
    task: str = "causal"
    exception: Exception | None = None


init_cases = [
    InitTestCase("new_run", False, False),
    InitTestCase("resume_no_ckpt", True, False),
    InitTestCase("resume_with_ckpt", True, True),
    InitTestCase(
        "resume_with_exception",
        True,
        False,
        "wrong",
        ValueError("Unknown task type: wrong. Use 'causal' or 'mapping'."),
    ),
]


@pytest.fixture
def config_tmp(tmp_path: Path, base_config: MockConfig) -> tuple[MockConfig, Path]:
    """Creates a temporary config and outputs directory."""
    return base_config, tmp_path


@pytest.mark.parametrize("case", init_cases, ids=lambda c: c.name)
def test_init(
    case: InitTestCase,
    config_tmp: tuple[MockConfig, Path],
    mocker: Any,
    mock_dependencies: dict[str, Any],
) -> None:
    """Tests the initialization logic of MambaTrainer including conditional resume paths."""
    base_config, tmp_path = config_tmp
    base_config.task = case.task  # type: ignore
    mock_resolve = mocker.patch.object(
        MambaTrainer, "_resolve_resume_path", return_value=tmp_path / "mock_dir"
    )
    mock_load = mocker.patch.object(
        MambaTrainer, "_load_config", return_value=base_config
    )
    mock_setup = mocker.patch.object(
        MambaTrainer, "_setup_trainer", return_value=mocker.Mock()
    )

    if case.has_checkpoint:
        mocker.patch(
            "src.engine.trainer.get_last_checkpoint", return_value="some/ckpt/path"
        )
    else:
        mocker.patch("src.engine.trainer.get_last_checkpoint", return_value=None)

    if case.exception:
        with pytest.raises(Exception) as exc_info:
            MambaTrainer(base_config, resume=case.resume)  # type: ignore
        assert str(exc_info.value) == str(case.exception)
        return

    trainer = MambaTrainer(base_config, resume=case.resume)  # type: ignore

    mock_dependencies["inject"].assert_called_once()
    mock_setup.assert_called_once()

    if case.resume:
        mock_resolve.assert_called_once_with(case.resume)
        assert trainer.resume is True
        if case.has_checkpoint:
            mock_load.assert_called_once_with(base_config, "some/ckpt/path")
        else:
            mock_load.assert_not_called()
    else:
        mock_resolve.assert_not_called()
        assert trainer.resume is False
        assert trainer.save_path == Path(base_config.save_path)


def test_setup_trainer(
    trainer_instance: MambaTrainer, mock_dependencies: dict[str, Any]
) -> None:
    """Tests the correct instantiation of Trainer and TrainingArguments using configured fields."""
    mock_args = mock_dependencies["args"]
    mock_trainer = mock_dependencies["trainer"]

    mock_args.assert_called_once()
    kwargs = mock_args.call_args.kwargs
    assert kwargs["output_dir"] == str(trainer_instance.save_path)
    assert kwargs["num_train_epochs"] == trainer_instance.cfg.scheduler_config.epochs
    assert kwargs["bf16"] is True

    mock_trainer.assert_called_once_with(
        model=trainer_instance.model,
        args=mock_args.return_value,
        train_dataset=trainer_instance.train_ds,
        eval_dataset=trainer_instance.eval_ds,
        compute_metrics=trainer_instance.compute_metrics,
        data_collator=trainer_instance.collator,
    )


@dataclass
class LoadConfigCase:
    """Data structure for checkpoint configuration synchronization tests."""

    name: str
    file_exists: bool
    file_content: dict[str, Any]
    expected_pad_token_id: int


load_cases = [
    LoadConfigCase(
        "file_exists_updates",
        True,
        {"pad_token_id": 999, "unknown_key": "ignore"},
        999,
    ),
    LoadConfigCase("file_missing_no_update", False, {}, 0),
]


@pytest.mark.parametrize("case", load_cases, ids=lambda c: c.name)
def test_load_config(
    case: LoadConfigCase, trainer_instance: MambaTrainer, tmp_path: Path
) -> None:
    """Tests synchronization of the Config object from a checkpoint project_config.json file."""
    ckpt_path = tmp_path / "ckpt"
    ckpt_path.mkdir()

    if case.file_exists:
        with open(ckpt_path / "project_config.json", "w") as f:
            json.dump(case.file_content, f)

    updated_config = trainer_instance._load_config(trainer_instance.cfg, str(ckpt_path))

    assert updated_config.pad_token_id == case.expected_pad_token_id


def test_save_config(trainer_instance: MambaTrainer, tmp_path: Path) -> None:
    """Tests serialization of the current Config object to a JSON file."""
    trainer_instance._save_config(tmp_path)
    file_path = tmp_path / "project_config.json"

    assert file_path.exists()
    with open(file_path) as f:
        data = json.load(f)

    assert data["max_len"] == trainer_instance.cfg.max_len
    assert data["save_path"] == trainer_instance.cfg.save_path
    assert data["pad_token_id"] == trainer_instance.cfg.pad_token_id


@dataclass
class ExplicitResumeCase:
    """Test cases for explicit resume path resolution."""

    name: str
    path_name: str
    should_exist: bool
    expect_error: bool

    def setup_filesystem(self, tmp_path: Path) -> Path:
        """Prepares the explicit target directory."""
        target = tmp_path / self.path_name
        if self.should_exist:
            target.mkdir(parents=True, exist_ok=True)
        return target


explicit_cases = [
    ExplicitResumeCase("explicit_exists", "valid_ckpt", True, False),
    ExplicitResumeCase("explicit_missing", "missing_ckpt", False, True),
]


@pytest.mark.parametrize("case", explicit_cases, ids=lambda c: c.name)
def test_resolve_explicit_resume_path(
    case: ExplicitResumeCase, trainer_instance: MambaTrainer, tmp_path: Path
) -> None:
    """Tests resolution when the user provides a specific path string."""
    target_path = case.setup_filesystem(tmp_path)

    if case.expect_error:
        with pytest.raises(FileNotFoundError):
            trainer_instance._resolve_resume_path(str(target_path))
    else:
        result = trainer_instance._resolve_resume_path(str(target_path))
        assert result == target_path


@dataclass
class AutoResumeCase:
    """Test cases for auto-detecting the latest run directory."""

    name: str
    use_spaces: bool
    create_base: bool
    subdirs: list[str]
    expected_dir: str | None
    expect_error: bool

    def setup_filesystem(self, base_path: Path) -> None:
        """Prepares the target directory structure and modification times."""
        if not self.create_base:
            return

        base_path.mkdir(parents=True, exist_ok=True)
        for idx, dir_name in enumerate(self.subdirs):
            sub_path = base_path / dir_name
            sub_path.mkdir()
            os.utime(sub_path, (100 + idx, 100 + idx))


auto_cases = [
    AutoResumeCase("base_missing", False, False, [], None, True),
    AutoResumeCase("no_subdirs", False, True, [], None, True),
    AutoResumeCase(
        "spaces_prefix",
        True,
        True,
        ["spaces_1", "spaces_2", "normal_1"],
        "spaces_2",
        False,
    ),
    AutoResumeCase(
        "normal_prefix",
        False,
        True,
        ["spaces_1", "normal_1", "normal_2"],
        "normal_2",
        False,
    ),
]


@pytest.mark.parametrize("case", auto_cases, ids=lambda c: c.name)
def test_resolve_auto_resume_path(
    case: AutoResumeCase, trainer_instance: MambaTrainer, tmp_path: Path
) -> None:
    """Tests auto-resolution based on configuration and filesystem state."""
    base_outputs = tmp_path / "outputs"
    case.setup_filesystem(base_outputs)

    trainer_instance.cfg.use_spaces = case.use_spaces
    trainer_instance.cfg.outputs_dir = base_outputs

    if case.expect_error:
        with pytest.raises(FileNotFoundError):
            trainer_instance._resolve_resume_path(True)
    else:
        result = trainer_instance._resolve_resume_path(True)
        assert case.expected_dir is not None
        assert result == base_outputs / case.expected_dir


@dataclass
class RunTestCase:
    """Data structure for testing execution runs based on checkpoint existence."""

    name: str
    resume: bool
    has_checkpoint: bool
    expect_train: bool


run_cases = [
    RunTestCase("resume_with_ckpt", True, True, True),
    RunTestCase("resume_missing_ckpt", True, False, False),
    RunTestCase("new_run", False, False, True),
]


@pytest.mark.parametrize("case", run_cases, ids=lambda c: c.name)
def test_run(case: RunTestCase, trainer_instance: MambaTrainer, mocker: Any) -> None:
    """Tests the main training run execution logic and saving callbacks."""
    trainer_instance.resume = case.resume
    mock_train = mocker.patch.object(trainer_instance.trainer, "train")
    mock_save_model = mocker.patch.object(trainer_instance.trainer, "save_model")
    mock_save_config = mocker.patch.object(trainer_instance, "_save_config")
    mock_logger_error = mocker.patch("src.engine.trainer.logger.error")

    if case.has_checkpoint:
        mocker.patch("src.engine.trainer.get_last_checkpoint", return_value="some/ckpt")
    else:
        mocker.patch("src.engine.trainer.get_last_checkpoint", return_value=None)

    result = trainer_instance.run()

    if not case.expect_train:
        assert result is None
        mock_logger_error.assert_called_once_with(
            "Resume requested but no checkpoint found."
        )
        mock_train.assert_not_called()
    else:
        if case.resume:
            mock_train.assert_called_once_with(resume_from_checkpoint="some/ckpt")
            assert mock_save_config.call_count == 1
        else:
            mock_train.assert_called_once_with(resume_from_checkpoint=None)
            assert mock_save_config.call_count == 2

        mock_save_model.assert_called_once_with(
            trainer_instance.save_path / "final_model"
        )

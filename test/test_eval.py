from unittest.mock import MagicMock
from src.eval import main

def test_eval_main_flow(mocker):
    mock_args = mocker.patch("src.eval.argparse.ArgumentParser.parse_args")
    mock_args.return_value = MagicMock(model_path="fake/path", spaces=True)

    mock_config_cls = mocker.patch("src.eval.Config")
    mock_config = mock_config_cls.return_value
    mock_config.tokenized_dir = MagicMock()

    mock_solver_cls = mocker.patch("src.eval.MambaCipherSolver")
    mock_load_ds = mocker.patch("src.eval.load_from_disk")

    main()

    mock_config.load_homophones.assert_called_once()
    mock_solver_cls.assert_called_once_with("fake/path", mock_config)
    mock_load_ds.assert_called_once()
    mock_solver_cls.return_value.evaluate.assert_called_once()

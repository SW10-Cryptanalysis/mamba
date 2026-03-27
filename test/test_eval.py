from unittest.mock import MagicMock
from src.eval import main

def test_eval_main_flow(mocker):
    # 1. Mock the Command Line Arguments
    mock_args = mocker.patch("src.eval.argparse.ArgumentParser.parse_args")
    mock_args.return_value = MagicMock(model_path="fake/path", spaces=True)

    # 2. Mock Config (and its internal path logic)
    mock_config_cls = mocker.patch("src.eval.Config")
    mock_config = mock_config_cls.return_value
    # Set the tokenized_dir so the / "Test" operation works on a Mock
    mock_config.tokenized_dir = MagicMock()

    # 3. Mock the Solver and Dataset loading
    mock_solver_cls = mocker.patch("src.eval.MambaCipherSolver")
    mock_load_ds = mocker.patch("src.eval.load_from_disk")

    # 4. Execute
    main()

    # 5. Assertions: Verify the chain of command
    mock_config.load_homophones.assert_called_once()
    mock_solver_cls.assert_called_once_with("fake/path", mock_config)
    mock_load_ds.assert_called_once()
    mock_solver_cls.return_value.evaluate.assert_called_once()

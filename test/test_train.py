from unittest.mock import MagicMock, patch
from src.train import main

class TestTrainScript:

    @patch("src.train.argparse.ArgumentParser.parse_args")
    @patch("src.train.Config")
    @patch("src.train.MambaTrainer")
    def test_train_main_flow(self, mock_trainer_cls, mock_config_cls, mock_parse_args):
        """Tests that train.py correctly wires config and triggers the trainer."""

        mock_args = MagicMock()
        mock_args.spaces = True
        mock_args.resume = True
        mock_parse_args.return_value = mock_args

        mock_config = mock_config_cls.return_value
        mock_trainer_instance = mock_trainer_cls.return_value

        main()

        assert mock_config.use_spaces is True
        mock_trainer_cls.assert_called_once_with(mock_config, resume=True)
        mock_trainer_instance.run.assert_called_once()

    @patch("src.train.argparse.ArgumentParser.parse_args")
    @patch("src.train.MambaTrainer")
    def test_train_resume_with_path(self, mock_trainer_cls, mock_parse_args):
        """Tests that a specific path string is passed correctly to the trainer."""

        mock_args = MagicMock()
        mock_args.spaces = False
        mock_args.resume = "./outputs/normal/run_2024"
        mock_parse_args.return_value = mock_args

        with patch("src.train.Config"):
            main()

            mock_trainer_cls.assert_called_once()
            args, kwargs = mock_trainer_cls.call_args
            assert kwargs["resume"] == "./outputs/normal/run_2024"

    def test_pytorch_alloc_conf_set(self):
        """Verifies the environment variable for memory management is set."""
        import os
        assert os.environ.get("PYTORCH_ALLOC_CONF") == "expandable_segments:True"

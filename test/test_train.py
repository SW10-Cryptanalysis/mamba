from unittest.mock import MagicMock, patch
from src.train import main

class TestTrainScript:

    @patch("src.train.argparse.ArgumentParser.parse_args")
    @patch("src.train.Config")
    @patch("src.train.MambaTrainer")
    def test_train_main_flow(self, mock_trainer_cls, mock_config_cls, mock_parse_args):
        """Tests that train.py correctly wires config and triggers the trainer."""

        # 1. Setup Mock Arguments (Simulating: python train.py --spaces --resume)
        mock_args = MagicMock()
        mock_args.spaces = True
        mock_args.resume = True
        mock_parse_args.return_value = mock_args

        # 2. Setup Mock Config
        mock_config = mock_config_cls.return_value

        # 3. Setup Mock Trainer Instance
        mock_trainer_instance = mock_trainer_cls.return_value

        # Execute
        main()

        # Assertions
        # Verify Config was updated with CLI args
        assert mock_config.use_spaces is True

        # Verify Trainer was initialized with the right arguments
        # resume comes from cmd_args.resume
        mock_trainer_cls.assert_called_once_with(mock_config, resume=True)

        # Verify the training loop was actually started
        mock_trainer_instance.run.assert_called_once()

    @patch("src.train.argparse.ArgumentParser.parse_args")
    @patch("src.train.MambaTrainer")
    def test_train_resume_with_path(self, mock_trainer_cls, mock_parse_args):
        """Tests that a specific path string is passed correctly to the trainer."""

        # Simulating: python train.py --resume ./outputs/normal/run_2024
        mock_args = MagicMock()
        mock_args.spaces = False
        mock_args.resume = "./outputs/normal/run_2024"
        mock_parse_args.return_value = mock_args

        with patch("src.train.Config"):
            main()

            # Ensure the specific path string reached the trainer
            mock_trainer_cls.assert_called_once()
            args, kwargs = mock_trainer_cls.call_args
            assert kwargs["resume"] == "./outputs/normal/run_2024"

    def test_pytorch_alloc_conf_set(self):
        """Verifies the environment variable for memory management is set."""
        import os
        assert os.environ.get("PYTORCH_ALLOC_CONF") == "expandable_segments:True"

import json
import time
import torch
from pathlib import Path
from torch.utils.data import Dataset
from src.config import Config
from transformers import Mamba2ForCausalLM
from src.utils.logging import get_logger

logger = get_logger(__name__)

class MambaCipherSolver:
    """A solver class for decoding ciphertexts using a Mamba2 language model.

    This class handles model loading, inference (solving), and batch evaluation
    against a dataset. It translates between raw token IDs and character strings
    based on a provided configuration.

    Attributes:
        config: Configuration object containing token IDs and offset mappings.
        model_path (Path): Directory path where the model weights are stored.
        device (str): The device being used for inference ('cuda' or 'cpu').
        model (Mamba2ForCausalLM): The loaded Mamba2 model instance.

    """

    def __init__(self, model_path: str, config: Config) -> None:
        """Initialize the solver and loads the Mamba2 model.

        Args:
            model_path (str): Path to the pretrained model directory.
            config: Configuration object with model and tokenizer parameters.

        """
        self.config = config
        self.model_path = Path(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model(model_path)

    def _load_model(self, path: str) -> Mamba2ForCausalLM:
        """Load the Mamba2 model from disk with appropriate precision.

        Args:
            path (str): Path to the model directory.

        Returns:
            Mamba2ForCausalLM: The loaded and evaluated model.

        """
        logger.info(f"Loading Mamba2 Solver from {path}...")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = Mamba2ForCausalLM.from_pretrained(
            path,
            torch_dtype=dtype,
            device_map="auto",
        )
        model.config.use_cache = True
        model.eval()
        return model

    @torch.no_grad()
    def solve(self, cipher_ids: list[int]) -> str:
        """Decode a single list of cipher IDs into a plaintext string.

        Constructs the prompt (BOS + Cipher + SEP), generates the completion,
        and slices the output to extract the predicted plaintext.

        Args:
            cipher_ids (list[int]): List of integer token IDs
                representing the ciphertext.

        Returns:
            str: The decoded plaintext string.

        """
        input_ids = [self.config.bos_token_id] + cipher_ids + [self.config.sep_token_id]
        input_tensor = torch.tensor([input_ids]).to(self.model.device)

        output_ids = self.model.generate(
            input_tensor,
            attention_mask=torch.ones_like(input_tensor),
            max_new_tokens=len(cipher_ids),
            min_new_tokens=len(cipher_ids),
            do_sample=False,
            use_cache=True,
            pad_token_id=self.config.pad_token_id,
            eos_token_id=self.config.eos_token_id,
        )

        pred_ids = output_ids[0][len(input_ids):].tolist()
        return self._decode_tokens(pred_ids)

    def _decode_tokens(self, ids: list[int]) -> str:
        """Map generated token IDs back to human-readable characters.

        Args:
            ids (list[int]): List of predicted token IDs from the model.

        Returns:
            str: The resulting plaintext string.

        """
        chars = []
        for idx in ids:
            if idx == self.config.space_token_id:
                chars.append("_" if self.config.use_spaces else " ")
            elif idx >= self.config.char_offset:
                chars.append(chr(idx - self.config.char_offset + ord("a")))
            elif idx == self.config.eos_token_id:
                break
        return "".join(chars)

    def _calculate_ser(self, true_plain: str, pred_plain: str) -> float:
        """Calculate the Symbol Error Rate (SER) between target and prediction.

        SER is calculated as (mismatches + length_difference) / target_length.

        Args:
            true_plain (str): The ground truth plaintext string.
            pred_plain (str): The string predicted by the model.

        Returns:
            float: The error rate between 0.0 and 1.0.

        """
        if not true_plain:
            return 1.0 if pred_plain else 0.0
        mismatches = sum(t != p for t, p in zip(true_plain, pred_plain, strict=False))
        length_diff = abs(len(true_plain) - len(pred_plain))
        return min((mismatches + length_diff) / len(true_plain), 1.0)

    def evaluate(self, dataset: Dataset) -> None:
        """Run the solver over an entire dataset and logs metrics to a JSONL file.

        Iterates through the dataset, extracts cipher IDs, generates predictions,
        calculates SER, and writes per-sample results to 'evaluation_results.jsonl'.

        Args:
            dataset: An iterable dataset containing 'input_ids' and 'raw_plaintext'.

        """
        output_log = self.model_path / "evaluation_results.jsonl"
        logger.info(f"Evaluating {len(dataset)} samples. Logging to {output_log}")

        total_ser, count = 0.0, 0

        for i, item in enumerate(dataset):
            all_ids = item["input_ids"]
            try:
                sep_idx = all_ids.index(self.config.sep_token_id)
                cipher_ids = all_ids[1:sep_idx]
            except ValueError:
                continue

            start = time.perf_counter()
            prediction = self.solve(cipher_ids)
            duration = time.perf_counter() - start

            ser = self._calculate_ser(item["raw_plaintext"], prediction)
            total_ser += ser
            count += 1

            # Save result
            result = {
                "idx": i,
                "target": item["raw_plaintext"],
                "pred": prediction,
                "ser": round(ser, 4),
                "time": round(duration, 4),
            }

            with open(output_log, "a") as f:
                f.write(json.dumps(result) + "\n")

            if i % 100 == 0:
                logger.info(f"Step {i} | Avg SER so far: {total_ser/count:.4f}")

        logger.info(f"Evaluation Complete. Final SER: {total_ser/count:.4f}")

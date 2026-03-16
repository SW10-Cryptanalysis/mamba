import torch
import torch.nn as nn
from mamba_ssm import Mamba2
from mamba_ssm.ops.triton.layer_norm import RMSNorm
from mamba_ssm.utils.generation import InferenceParams
from src.config import Config

class MambaModel(nn.Module):
    """Mamba-based sequence model for cipher decryption.

    This model uses the Mamba2 architecture to process sequences of cipher
    homophones and predict the corresponding plaintext characters.

    Attributes:
        char_offset (int): The index where plaintext characters begin in the vocab.
        embedding (nn.Embedding): Learnt embeddings for the input tokens.
        layers (nn.ModuleList): A list of Mamba2 layers with RMSNorm.
        norm_f (RMSNorm): Final normalization layer before the head.
        lm_head (nn.Linear): Linear layer mapping hidden states to vocabulary logits.

    """

    def __init__(
        self,
        vocab_size: int,
        char_offset: int,
        config: Config,
    ) -> None:
        """Initialize the MambaModel.

        Args:
            vocab_size: Total number of tokens in the vocabulary.
            char_offset: Offset used to separate cipher and plain tokens.
            config: Configuration object with model hyperparameters.

        """
        super().__init__()
        self.char_offset = char_offset
        self.embedding = nn.Embedding(vocab_size, config.d_model)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm": RMSNorm(config.d_model),
                "mixer": Mamba2(
                    d_model=config.d_model,
                    d_state=config.d_state,
                    d_conv=config.d_conv,
                    expand=config.expand,
                ),
            })
            for _ in range(config.n_layers)
        ])

        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor, inference_params: InferenceParams = None) -> torch.Tensor:
        """Perform a forward pass through the network.

        Args:
            x: Input tensor of token IDs with shape (batch_size, seq_len).

        Returns:
            Logits tensor with shape (batch_size, seq_len, vocab_size).

        """
        x = self.embedding(x)
        for layer in self.layers:
            residual = x
            x = layer["norm"](x)
            x = layer["mixer"](x, inference_params=inference_params) + residual

        x = self.norm_f(x)
        return self.lm_head(x)

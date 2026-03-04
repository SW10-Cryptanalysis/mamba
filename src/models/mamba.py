import torch
import torch.nn as nn
from mamba_ssm import Mamba2
from mamba_ssm.ops.triton.layer_norm import RMSNorm
from src.config import Config

class MambaModel(nn.Module):
    """MambaCipherSolver model."""

    def __init__(
        self,
        vocab_size: int,
        char_offset: int,
        config: Config,
    ) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            residual = x
            x = layer["norm"](x)
            x = layer["mixer"](x) + residual

        x = self.norm_f(x)
        return self.lm_head(x)

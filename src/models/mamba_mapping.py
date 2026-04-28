from dataclasses import asdict
import torch
import torch.nn as nn
from transformers import Mamba2Config, Mamba2Model
from transformers.modeling_outputs import SequenceClassifierOutput
from src.config import MambaConfig
from src.utils.logging import get_logger
from typing import Any

logger = get_logger(__name__)


class Mamba2ForMapping(nn.Module):
    """Custom Mamba2 model for mapping prediction (Token Classification).

    This model passes the sequence through the causal Mamba2 backbone,
    pools the hidden states for each unique cipher symbol, and passes
    them through a linear classification head to predict the English letter.

    Attributes:
        config (Mamba2Config): The Mamba2 configuration object.
        num_labels (int): The number of labels in the classification task.
        mamba2 (Mamba2Model): The Mamba2 backbone model.
        classifier (nn.Linear): The linear classification head.

    """

    def __init__(self, config: Mamba2Config, num_labels: int = 26) -> None:
        """Initialize the Mamba2 model with the provided configuration.

        Args:
            config (Mamba2Config): The Mamba2 configuration object.
            num_labels (int, optional): The number of labels in the classification task.
                Defaults to 26.

        """
        super().__init__()
        self.num_labels = num_labels
        self.config = config

        self.mamba2 = Mamba2Model(config)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: dict[str, Any],
    ) -> SequenceClassifierOutput:
        """Forward pass through the Mamba2 model.

        This method processes the input sequence through the causal backbone,
        pools the hidden states for each unique cipher symbol, and passes
        them through a linear classification head to predict the English letter.

        Args:
            input_ids (torch.Tensor): The input sequence of token IDs.
            labels (torch.Tensor | None, optional): The target labels for the
                classification task. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the causal backbone.

        Returns:
            SequenceClassifierOutput: The output of the classification head.

        """
        outputs = self.mamba2(input_ids=input_ids, **kwargs)
        hidden_states = outputs.last_hidden_state

        batch_size = input_ids.size(0)
        pooled_logits_list = []
        aligned_labels_list = []

        for b in range(batch_size):
            seq_ids = input_ids[b]
            seq_hidden = hidden_states[b]

            unique_symbols = torch.unique(seq_ids)
            unique_symbols = unique_symbols[unique_symbols > 0]

            pooled_vectors = [
                seq_hidden[seq_ids == symbol].mean(dim=0) for symbol in unique_symbols
            ]

            pooled_tensor = torch.stack(pooled_vectors)

            logits = self.classifier(pooled_tensor)
            pooled_logits_list.append(logits)

            if labels is not None:
                num_symbols = logits.size(0)
                aligned_labels_list.append(labels[b][:num_symbols])

        final_logits = torch.cat(pooled_logits_list, dim=0)

        loss = None
        if labels is not None and aligned_labels_list:
            final_labels = torch.cat(aligned_labels_list, dim=0)

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(final_logits, final_labels.to(final_logits.device))

        return SequenceClassifierOutput(
            loss=loss,
            logits=final_logits,  # type: ignore
            hidden_states=outputs.hidden_states,
        )


def get_mapping_model(config: MambaConfig) -> Mamba2ForMapping:
    """Initialize a Mamba2 Mapping model with parameters defined in the config.

    Args:
        config (MambaConfig): The Mamba2 configuration object.

    Returns:
        Mamba2ForMapping: The initialized Mamba2 Mapping model.

    """
    m_dict = asdict(config)

    mamba2_config = Mamba2Config(**m_dict)
    mamba2_config.torch_dtype = torch.bfloat16

    model = Mamba2ForMapping(mamba2_config, num_labels=26)

    logger.info("Mamba2 Mapping Model loaded!")
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters:       {num_params:,}")

    return model

import sys
from unittest.mock import MagicMock

# Mock mamba & causalconv1d modules for testing

modules_to_mock = [
    "mamba_ssm",
    "mamba_ssm.utils",
    "mamba_ssm.utils.generation",
    "mamba_ssm.ops",
    "mamba_ssm.ops.triton",
    "mamba_ssm.ops.triton.ssd_combined",
    "mamba_ssm.ops.triton.selective_state_update",
    "mamba_ssm.ops.triton.layer_norm",
    "mamba_ssm.ops.selective_scan_interface",
    "causal_conv1d",
    "causal_conv1d.causal_conv1d_interface",
    "transformers",
    "transformers.trainer_utils",
    "transformers.modeling_outputs",
    "datasets"
]

for mod in modules_to_mock:
    sys.modules[mod] = MagicMock()

sys.modules["mamba_ssm.ops.triton.layer_norm"].RMSNorm = MagicMock # type: ignore
sys.modules["mamba_ssm"].Mamba2 = MagicMock # type: ignore

mamba2_modeling_mock = MagicMock()
mamba2_modeling_mock.is_fast_path_available = True
sys.modules["transformers.models.mamba2.modeling_mamba2"] = mamba2_modeling_mock

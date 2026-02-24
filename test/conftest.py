import sys
from unittest.mock import MagicMock

# Mock mamba & causalconv1d modules for testing

modules_to_mock = [
    "mamba_ssm",
    "mamba_ssm.ops",
    "mamba_ssm.ops.triton",
    "mamba_ssm.ops.triton.layer_norm",
    "mamba_ssm.ops.selective_scan_interface",
    "causal_conv1d",
    "causal_conv1d.causal_conv1d_interface",
]

for mod in modules_to_mock:
    sys.modules[mod] = MagicMock()

sys.modules["mamba_ssm.ops.triton.layer_norm"].RMSNorm = MagicMock
sys.modules["mamba_ssm"].Mamba2 = MagicMock

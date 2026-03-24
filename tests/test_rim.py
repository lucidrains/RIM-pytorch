import pytest
import torch

param = pytest.mark.parametrize

@param('causal', (False, True))
@param('readout', (False, True))
def test_rim(
    causal,
    readout
):
    from RIM_pytorch import RIM
    from RIM_pytorch.depth_less_transformer import DepthlessTransformer

    model = DepthlessTransformer(
        512,
        causal = causal,
        num_blocks = 6,
        num_tokens = 256 if readout else None,
    )

    x = torch.randn(1, 1024, 512)
    out = model(x)

    if readout:
        logits = out
        assert logits.shape == (1, 1024, 256)

    else:
        pooled_messages = out
        assert pooled_messages.shape == (1, 1024, 512)

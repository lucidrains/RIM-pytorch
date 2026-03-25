import pytest
import torch

param = pytest.mark.parametrize

@param('causal', (False, True))
@param('readout', (False, True))
@param('use_pope', (False, True))
def test_rim(
    causal,
    readout,
    use_pope
):
    from RIM_pytorch import RIM
    from RIM_pytorch.depth_less_transformer import DepthlessTransformer

    model = DepthlessTransformer(
        32,
        causal = causal,
        use_pope = use_pope,
        num_blocks = 6,
        num_tokens = 16 if readout else None,
    )

    x = torch.randn(1, 7, 32)
    out = model(x)

    if readout:
        logits = out
        assert logits.shape == (1, 7, 16)

    else:
        pooled_messages = out
        assert pooled_messages.shape == (1, 7, 32)

def test_ensemble_message_passing_mlp():
    from RIM_pytorch.depth_less_transformer import EnsemblesWithMessagePassing, Attention
    from x_mlps_pytorch import MLP

    dim = 64
    heads = 4
    dim_head = 16
    ensemble_size = 2

    mlp = MLP(dim, dim * 2, dim)

    voting_attn = Attention(dim, dim_head = dim_head, heads = heads)

    model = EnsemblesWithMessagePassing(
        modules = dict(mlp = mlp),
        ensemble_size = ensemble_size,
        voting_attn = voting_attn,
        num_message_exchanges = 2
    )

    x = torch.randn(3, dim)

    tokens = model(x, repeat_input_for_ensemble = True)

    assert tokens.shape == (ensemble_size, 3, dim)

def test_nested_ensemble_message_passing_mlp():
    from RIM_pytorch.depth_less_transformer import EnsemblesWithMessagePassing, Attention
    from x_mlps_pytorch import MLP

    dim = 64
    heads = 4
    dim_head = 16

    mlp = MLP(dim, dim * 2, dim)

    inner_model = EnsemblesWithMessagePassing(
        modules = dict(mlp = mlp),
        ensemble_size = 4,
        voting_attn = Attention(dim, dim_head = dim_head, heads = heads),
        num_message_exchanges = 2
    )

    outer_model = EnsemblesWithMessagePassing(
        modules = dict(inner = inner_model),
        ensemble_size = 4,
        voting_attn = Attention(dim, dim_head = dim_head, heads = heads),
        num_message_exchanges = 2
    )

    x = torch.randn(4, 4, 3, dim)

    tokens = outer_model(x)

    assert tokens.shape == x.shape

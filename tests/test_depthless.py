import torch
from RIM_pytorch.depth_less_transformer import DepthlessTransformer

def test_depthless_parallel():
    model = DepthlessTransformer(
        dim = 256,
        num_tokens = 1000,
        num_blocks = 3,
        num_message_exchanges = 3
    )

    tokens = torch.randint(0, 1000, (2, 32))
    logits = model(tokens)

    assert logits.shape == (2, 32, 1000)


def test_depthless_sequential_routing():
    model = DepthlessTransformer(
        dim = 256,
        num_tokens = 1000,
        num_blocks = 3,
        num_message_exchanges = 6 # overridden by schedule
    )

    routing_schedule = (
        (('attn', (0,)),),
        (('ff', (0,)),),
        (('attn', (1,)),),
        (('ff', (1,)),),
        (('attn', (2,)),),
        (('ff', (2,)),),
    )

    tokens = torch.randint(0, 1000, (2, 32))
    logits = model(tokens, routing_schedule = routing_schedule)

    assert logits.shape == (2, 32, 1000)


def test_depthless_rotational_routing():
    model = DepthlessTransformer(
        dim = 256,
        num_tokens = 1000,
        num_blocks = 3,
        num_message_exchanges = 3 # overridden by schedule
    )

    routing_schedule = (
        (('attn', (0, 1)), ('ff', (2,))),
        (('attn', (2,)), ('ff', (0, 1))),
    )

    tokens = torch.randint(0, 1000, (2, 32))
    logits = model(tokens, routing_schedule = routing_schedule)

    assert logits.shape == (2, 32, 1000)

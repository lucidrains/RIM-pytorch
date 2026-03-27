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


def test_depthless_int_routing_schedule():
    model = DepthlessTransformer(
        dim = 256,
        num_tokens = 1000,
        num_blocks = 3,
        num_message_exchanges = 6
    )

    routing_schedule = (
        (('attn', 0),),
        (('ff', 0),),
        (('attn', 1),),
        (('ff', 1),),
        (('attn', 2),),
        (('ff', 2),),
    )

    tokens = torch.randint(0, 1000, (2, 32))
    logits = model(tokens, routing_schedule = routing_schedule)

    assert logits.shape == (2, 32, 1000)

def test_depthless_init_routing_schedule():
    routing_schedule = (
        (('attn', (0,)),),
        (('ff', (0,)),),
    )

    model = DepthlessTransformer(
        dim = 256,
        num_tokens = 1000,
        num_blocks = 3,
        routing_schedule = routing_schedule
    )

    tokens = torch.randint(0, 1000, (2, 32))

    # Use init routing schedule (length 2 expected)
    logits, messages = model(tokens, return_messages=True)
    assert len(messages) == 3 # init + round 1 + round 2
    assert logits.shape == (2, 32, 1000)

    # Override routing schedule (length 1 expected)
    override_schedule = (
        (('attn', 1),),
    )
    logits2, messages2 = model(tokens, return_messages=True, routing_schedule=override_schedule)
    assert len(messages2) == 2 # init + round 1
    assert logits2.shape == (2, 32, 1000)

def test_depthless_implicit_all_indices_routing_schedule():
    routing_schedule = (
        ('attn', 'ff'),
        (('attn', None), ('ff', None)),
    )

    model = DepthlessTransformer(
        dim = 256,
        num_tokens = 1000,
        num_blocks = 3,
        routing_schedule = routing_schedule
    )

    tokens = torch.randint(0, 1000, (2, 32))

    logits, messages = model(tokens, return_messages=True)
    assert len(messages) == 5 # init + attn(R1) + ff(R1) + attn(R2) + ff(R2)
    assert messages[1].shape == (3, 2, 32, 256) # 3 attn blocks
    assert logits.shape == (2, 32, 1000)

def test_implicit_single_module_inference():
    from RIM_pytorch.depth_less_transformer import EnsemblesWithMessagePassing, Attention

    attn = Attention(dim = 256)

    model = EnsemblesWithMessagePassing(
        dim = 256,
        modules = attn,
        ensemble_size = 3,
        num_message_exchanges = 2
    )

    routing_schedule = (
        ((0, 1),),
        (2,),
    )

    tokens = torch.randn(2, 32, 256)

    messages = model(
        tokens,
        repeat_input_for_ensemble = True,
        return_all_messages = True,
        routing_schedule = routing_schedule
    )

    assert len(messages) == 3
    assert messages[1].shape == (2, 2, 32, 256)
    assert messages[2].shape == (1, 2, 32, 256)

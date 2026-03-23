from __future__ import annotations
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear
from torch.func import vmap, functional_call

from torch_einops_utils import pack_with_inverse

from einops import einsum, repeat, rearrange, pack
from einops.layers.torch import Rearrange, Reduce

# einstein notation

# m - messages
# l - bLocks
# b - batch
# n - sequence
# d - feature dimension
# i, j - source and target sequence for attention
# h - attention heads

# constants

LinearNoBias = partial(Linear, bias = False)

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.norm = nn.RMSNorm(dim)

        self.to_q = LinearNoBias(dim, dim_inner)
        self.to_kv = LinearNoBias(dim, dim_inner * 2)

        self.to_out = LinearNoBias(dim_inner, dim)

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)

        self.merge_heads = Rearrange('b h n d -> b n (h d)')

    def forward(
        self,
        tokens,
        context = None
    ):
        tokens, inverse_pack = pack_with_inverse(tokens, '* n d')
        tokens = self.norm(tokens)

        if exists(context):
            context, _ = pack_with_inverse(context, '* n d')
        else:
            context = tokens

        q, k, v = (self.to_q(tokens), *self.to_kv(context).chunk(2, dim = -1))

        q, k, v = map(self.split_heads, (q, k, v))

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        attn = sim.softmax(dim = -1)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        out = self.merge_heads(out)
        out = self.to_out(out)
        return inverse_pack(out)

# swiglu ff - Shazeer et al

class Feedforward(Module):
    def __init__(
        self,
        dim,
        expansion_factor = 4.
    ):
        super().__init__()
        dim_inner = int(dim * expansion_factor * 2 / 3)

        self.norm = nn.RMSNorm(dim)
        self.keys = Linear(dim, dim_inner * 2)
        self.values = Linear(dim_inner, dim)

    def forward(
        self,
        tokens
    ):
        queries = self.norm(tokens)
        sim, gates = self.keys(queries).chunk(2, dim = -1)
        sim = sim * F.gelu(gates)
        return self.values(sim)

# classes

class DepthlessTransformer(Module):
    def __init__(
        self,
        dim,
        num_blocks = 6,
        num_message_exchanges = 6,
        dim_head = 64,
        heads = 8,
        ff_expansion_factor = 4.
    ):
        super().__init__()

        self.num_message_exchanges = num_message_exchanges

        self.num_blocks = num_blocks
        repeat_blocks = Reduce('... -> blocks ...', 'repeat', blocks = num_blocks)
        self.repeat_blocks = repeat_blocks

        # define attention and feedforward

        self.attn = Attention(dim, dim_head = dim_head, heads = heads)
        self.ff = Feedforward(dim, ff_expansion_factor)

        # the attention residual, or just putting together the information coming from various recurrent modules

        self.attn_residual = Attention(dim, dim_head = dim_head, heads = heads)

        # functional forwards

        def attn_forward(params, inputs):
            return functional_call(self.attn, params, inputs)

        def ff_forward(params, inputs):
            return functional_call(self.ff, params, inputs)

        self.attn_parameters = {name: repeat_blocks(param) for name, param in self.attn.named_parameters()}
        self.ff_parameters = {name: repeat_blocks(param) for name, param in self.ff.named_parameters()}

        # vmap over blocks dimension to call all block at once across tokens

        self.attn_forward = vmap(attn_forward, in_dims = (0, 0))
        self.ff_forward = vmap(ff_forward, in_dims = (0, 0))

    def forward(
        self,
        tokens
    ):
        batch, blocks = tokens.shape[0], self.num_blocks

        tokens = self.repeat_blocks(tokens) # (blocks b n d)

        # reframed as recurrent processing of tokens with message passing (attention residual)

        messages = [tokens]

        for i in range(self.num_message_exchanges):
            attended = self.attn_forward(self.attn_parameters, tokens)
            retrieved_memories = self.ff_forward(self.ff_parameters, tokens)

            # add to processed messages

            messages.extend([attended, retrieved_memories])

            # then we just do attention pooling / residual for next round
            # will use the initial messages coming in as the queries, all products of all the blocks become messages

            packed_messages, _ = pack(messages, '* b n d') # (message blocks) packed

            packed_messages = repeat(packed_messages, 'm b n d -> (b l) n m d', l = blocks)

            message_queries, inverse_pack_blocks = pack_with_inverse(tokens, '* n d') # (m b n d)
            message_queries = rearrange(message_queries, '... n d -> ... n 1 d')

            # each message producer attends to all messages (and their history) by all other producers

            pooled_messages = self.attn_residual(message_queries, packed_messages)

            pooled_messages = inverse_pack_blocks(pooled_messages, '* n one d')

            pooled_messages = rearrange(pooled_messages, 'l b n 1 d -> l b n d')

            # keep iterating

            tokens = pooled_messages

        return tokens

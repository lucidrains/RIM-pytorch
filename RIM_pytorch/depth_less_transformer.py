from __future__ import annotations
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, Linear, Parameter, ParameterList
from torch.func import vmap, functional_call

from torch_einops_utils import pack_with_inverse
from PoPE_pytorch import PoPE, flash_attn_with_pope

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

# types

RoutingIndices = tuple[int, ...] | int | None
RoutingConfig = tuple[str, RoutingIndices] | str
RoutingRound = tuple[RoutingConfig, ...]
RoutingSchedule = tuple[RoutingRound, ...]

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
        heads = 8,
        causal = False,
        key_rmsnorm = False,
        dropout = 0.,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads
        self.causal = causal

        self.norm = nn.RMSNorm(dim)
        self.maybe_key_norm = nn.RMSNorm(dim_head) if key_rmsnorm else nn.Identity()

        self.to_q = LinearNoBias(dim, dim_inner)
        self.to_kv = LinearNoBias(dim, dim_inner * 2)

        self.to_out = LinearNoBias(dim_inner, dim)

        self.to_gates = nn.Sequential(LinearNoBias(dim, heads), Rearrange('... n h -> ... h n 1'))

        self.dropout_prob = dropout
        self.dropout = nn.Dropout(dropout)

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)

        self.merge_heads = Rearrange('b h n d -> b n (h d)')

    def forward(
        self,
        tokens,
        context = None,
        pos_emb = None
    ):
        device = tokens.device
        tokens, inverse_pack = pack_with_inverse(tokens, '* n d')
        tokens = self.norm(tokens)

        if exists(context):
            context, _ = pack_with_inverse(context, '* n d')
        else:
            context = tokens

        q, k, v = (self.to_q(tokens), *self.to_kv(context).chunk(2, dim = -1))

        q, k, v = map(self.split_heads, (q, k, v))

        k = self.maybe_key_norm(k)

        if exists(pos_emb):
            out = flash_attn_with_pope(
                q, k, v,
                pos_emb = pos_emb,
                causal = self.causal,
                dropout = self.dropout_prob,
                softmax_scale = self.scale
            )
        else:
            sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

            sim = sim * self.scale

            if self.causal:
                i, j = sim.shape[-2:]
                causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
                sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

            attn = sim.softmax(dim = -1)

            attn = self.dropout(attn)

            out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        out = self.to_gates(tokens).sigmoid() * out

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
        sim = sim * F.silu(gates)
        return self.values(sim)

# ensemble

class Ensemble(Module):
    def __init__(
        self,
        net: Module,
        ensemble_size: int
    ):
        super().__init__()
        repeat_blocks = Reduce('... -> l ...', 'repeat', l = ensemble_size)

        named_params = dict(net.named_parameters())

        # avoid the issue with period in the parameter names

        self.param_names = named_params.keys()
        self.net_parameters = ParameterList([Parameter(repeat_blocks(param).clone()) for param in named_params.values()])

        self.init_()

        # vmapping

        def net_forward(params, tokens, *args, **kwargs):
            return functional_call(net, params, args = (tokens, *args), kwargs = kwargs)

        self.net_forward = vmap(net_forward, in_dims = 0, randomness = 'different')

    @torch.no_grad()
    def init_(self):
        for name, param in self.parameters.items():
            if 'norm' in name:
                nn.init.ones_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
            elif 'weight' in name:
                fan_in = param.shape[-1]
                bound = fan_in ** -0.5
                nn.init.uniform_(param, -bound, bound)

    @property
    def parameters(self):
        return dict(zip(self.param_names, self.net_parameters))

    def forward(
        self,
        tokens,
        indices: tuple[int, ...] | int | None = None,
        *args,
        **kwargs
    ):
        params = self.parameters

        if exists(indices):
            if isinstance(indices, int):
                indices = (indices,)

            indices = list(indices)
            params = {k: v[indices] for k, v in params.items()}
            tokens = tokens[indices]

        return self.net_forward(params, tokens, *args, **kwargs)

# ensembles with message passing

class EnsemblesWithMessagePassing(Module):
    def __init__(
        self,
        modules: dict[str, Module] | Module,
        ensemble_size: int,
        *,
        dim: int | None = None,
        voting_attn: Module | None = None,
        voting_attn_kwargs: dict = dict(dim_head = 64, heads = 8),
        num_message_exchanges: int = 1,
        routing_schedule: RoutingSchedule | None = None
    ):
        super().__init__()
        self.num_message_exchanges = num_message_exchanges
        self.ensemble_size = ensemble_size
        self.routing_schedule = routing_schedule

        if isinstance(modules, Module):
            modules = dict(module = modules)

        self.ensembles = nn.ModuleDict({
            name: Ensemble(module, ensemble_size) for name, module in modules.items()
        })

        assert isinstance(voting_attn, Module) ^ exists(dim), 'either voting_attn is passed in as a Module or dim is passed in'

        if not isinstance(voting_attn, Module):
            voting_attn = Attention(dim, key_rmsnorm = True, **voting_attn_kwargs)

        self.voting_attn = voting_attn

    def forward(
        self,
        tokens, # (b ...) or (l b ...)
        module_kwargs: dict[str, dict] | None = None,
        repeat_input_for_ensemble: bool = False,
        return_all_messages: bool = False,
        num_message_exchanges: int | None = None,
        routing_schedule: RoutingSchedule | None = None
    ): # (l b ...)

        routing_schedule = default(routing_schedule, self.routing_schedule)

        if exists(routing_schedule):
            num_message_exchanges = len(routing_schedule)
        else:
            num_message_exchanges = default(num_message_exchanges, self.num_message_exchanges)

        if repeat_input_for_ensemble:
            tokens = repeat(tokens, '... -> l ...', l = self.ensemble_size)

        module_kwargs = default(module_kwargs, dict())

        if len(module_kwargs) > 0:
            first_key = next(iter(module_kwargs.keys()))
            if first_key not in self.ensembles:
                assert len(self.ensembles) == 1, 'module_kwargs must be a dictionary with module names as keys if there are multiple modules in the ensemble'
                default_name = next(iter(self.ensembles.keys()))
                module_kwargs = {default_name: module_kwargs}

        messages = [tokens]
        blocks = self.ensemble_size

        # reframed as recurrent processing of tokens with message passing (attention residual)

        for count in range(1, num_message_exchanges + 1):
            is_last = count == num_message_exchanges

            # collect messages from all ensembles

            active_modules = routing_schedule[count - 1] if exists(routing_schedule) else tuple((name, None) for name in self.ensembles.keys())

            for config in active_modules:
                if isinstance(config, str) and config in self.ensembles:
                    mod_name = config
                    indices = None
                elif isinstance(config, tuple) and len(config) == 2 and isinstance(config[0], str) and config[0] in self.ensembles:
                    mod_name, indices = config
                else:
                    assert len(self.ensembles) == 1, 'if passing indices directly or relying on string inference in routing_schedule, there must only be one module in the ensemble'
                    mod_name = next(iter(self.ensembles.keys()))
                    indices = config

                if exists(indices):
                    if isinstance(indices, int):
                        indices = (indices,)

                    if len(indices) == 0:
                        continue

                ensemble = self.ensembles[mod_name]
                kwargs = module_kwargs.get(mod_name, dict())
                out = ensemble(tokens, indices = indices, **kwargs)
                messages.append(out)

            if is_last and return_all_messages:
                continue

            # then we just do attention pooling (attention 'residual') for next round
            # will use the initial messages coming in as the queries, all products of all the blocks become messages - voting phase

            flat_messages = torch.cat(messages, dim = 0)
            all_messages = repeat(flat_messages, 'm b ... d -> (lq b) ... m d', lq = blocks)

            message_queries = rearrange(tokens, 'l b ... d -> (l b) ... 1 d')

            # each message producer attends to all messages (and their history) by all other producers

            pooled_messages = self.voting_attn(message_queries, all_messages)

            pooled_messages = rearrange(pooled_messages, '(l b) ... 1 d -> l b ... d', l = blocks)

            # keep iterating

            tokens = pooled_messages

        if not return_all_messages:
            return tokens

        return messages

# classes

class DepthlessTransformer(Module):
    def __init__(
        self,
        dim,
        num_blocks = 6,
        num_message_exchanges = 6,
        dim_head = 64,
        heads = 8,
        causal = False,
        ff_expansion_factor = 4.,
        num_tokens = None,
        use_pope = False,
        routing_schedule: RoutingSchedule | None = None
    ):
        super().__init__()

        self.num_message_exchanges = num_message_exchanges

        self.num_blocks = num_blocks

        self.use_pope = use_pope
        if use_pope:
            self.pope = PoPE(dim = dim_head, heads = heads)

        # define attention and feedforward

        attn = Attention(dim, causal = causal, dim_head = dim_head, heads = heads)
        ff = Feedforward(dim, ff_expansion_factor)

        # token embedding

        self.token_emb = nn.Embedding(num_tokens, dim) if exists(num_tokens) else None

        # ensembles with message passing

        self.ensembles_with_message_passing = EnsemblesWithMessagePassing(
            modules = dict(
                attn = attn,
                ff = ff
            ),
            ensemble_size = num_blocks,
            dim = dim,
            voting_attn_kwargs = dict(dim_head = dim_head, heads = heads),
            num_message_exchanges = num_message_exchanges,
            routing_schedule = routing_schedule
        )

        # the attention residual, or just putting together the information coming from various recurrent modules

        self.attn_residual = Attention(dim, key_rmsnorm = True, dim_head = dim_head, heads = heads)

        # readout

        self.attn_pool_query = nn.Parameter(torch.randn(dim) * 1e-2)
        self.readout = nn.Sequential(nn.RMSNorm(dim), LinearNoBias(dim, num_tokens)) if exists(num_tokens) else None

    def forward(
        self,
        tokens,
        return_messages = False,
        num_message_exchanges: int | None = None,
        routing_schedule: RoutingSchedule | None = None
    ):
        if exists(self.token_emb):
            tokens = self.token_emb(tokens)

        batch, seq_len, blocks = *tokens.shape[:2], self.num_blocks

        # positions forwarded to attn ensemble

        module_kwargs = dict()
        if self.use_pope:
            pos_emb = self.pope(seq_len)
            module_kwargs = dict(attn = dict(pos_emb = pos_emb))

        # message passing

        messages = self.ensembles_with_message_passing(
            tokens,
            module_kwargs = module_kwargs,
            repeat_input_for_ensemble = True,
            return_all_messages = True,
            num_message_exchanges = num_message_exchanges,
            routing_schedule = routing_schedule
        )

        # the readout itself is just another message producer

        queries = repeat(self.attn_pool_query, 'd -> b n 1 d', b = batch, n = seq_len)

        flat_messages = torch.cat(messages, dim = 0)
        all_messages = rearrange(flat_messages, 'm b n d -> b n m d')

        readout_input = self.attn_residual(queries, all_messages)

        readout_input = rearrange(readout_input, 'b n 1 d -> b n d')

        if not exists(self.readout):
            return readout_input

        logits = self.readout(readout_input)

        if not return_messages:
            return logits

        return logits, messages

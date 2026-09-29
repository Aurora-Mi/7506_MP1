"""Student GPT: RoPE + RMSNorm + SwiGLU, with config switches for ablation."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        x_f = x.float()
        scale = x_f.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x_f * scale * self.weight.float()).type_as(x)


class SwiGLU(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.gate_up = nn.Linear(width, 2 * hidden)
        self.down = nn.Linear(hidden, width)

    def forward(self, x):
        gate, value = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


class TokenRoutedMoE(nn.Module):
    """Sparse FFN experts with deterministic, strictly causal token routing.

    Routing by the currently observed token avoids router collapse and adds no
    auxiliary loss. Only one expert is evaluated for each token position.
    """
    def __init__(self, width, hidden, experts):
        super().__init__()
        self.experts = nn.ModuleList([SwiGLU(width, hidden) for _ in range(experts)])

    def forward(self, x, routes):
        shape = x.shape
        flat_x = x.reshape(-1, shape[-1])
        flat_routes = routes.reshape(-1)
        output = torch.zeros_like(flat_x)
        for expert_id, expert in enumerate(self.experts):
            indices = (flat_routes == expert_id).nonzero(as_tuple=False).flatten()
            if indices.numel() > 0:
                selected = flat_x.index_select(0, indices)
                # Autocast may produce BF16 expert outputs while the residual
                # stream (and therefore this buffer) remains FP32.
                expert_output = expert(selected).to(dtype=output.dtype)
                output.index_copy_(0, indices, expert_output)
        return output.view(shape)


def rope_cache(context, head_dim, base=10000.0):
    """Build the fixed-context RoPE table once instead of once per block/batch."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.arange(context, dtype=torch.float32)[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos()[None, None], angles.sin()[None, None]


def apply_rope(q, k, cos, sin):
    head_dim = q.shape[-1]
    length = q.shape[2]
    cos = cos[:, :, :length].to(dtype=q.dtype)
    sin = sin[:, :, :length].to(dtype=q.dtype)

    def rotate(x):
        x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2 :]
        return x * cos + torch.cat((-x2, x1), dim=-1) * sin
    return rotate(q), rotate(k)


class Block(nn.Module):
    def __init__(self, width, heads, rmsnorm, swiglu, dropout=0.0,
                 ffn_multiplier=None, moe_experts=1,
                 attention_dropout=None, residual_dropout=None):
        super().__init__()
        self.heads = heads
        self.attention_dropout = dropout if attention_dropout is None else attention_dropout
        self.residual_dropout = dropout if residual_dropout is None else residual_dropout
        make_norm = RMSNorm if rmsnorm else nn.LayerNorm
        self.norm1, self.norm2 = make_norm(width), make_norm(width)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        if swiglu:
            multiplier = 8 / 3 if ffn_multiplier is None else float(ffn_multiplier)
            hidden = max(8, (int(multiplier * width) + 7) // 8 * 8)
            self.mlp = (TokenRoutedMoE(width, hidden, moe_experts)
                        if moe_experts > 1 else SwiGLU(width, hidden))
        else:
            self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x, rope, routes=None):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        if rope is not None:
            q, k = apply_rope(q, k, *rope)
        attended = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attention_dropout if self.training else 0.0, is_causal=True
        )
        attention_out = self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        x = x + F.dropout(attention_out, self.residual_dropout, self.training)
        normalized = self.norm2(x)
        mlp_out = self.mlp(normalized, routes) if isinstance(self.mlp, TokenRoutedMoE) else self.mlp(normalized)
        return x + F.dropout(mlp_out, self.residual_dropout, self.training)


class StudentGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        self.use_rope = bool(config.get('rope', True))
        self.cache_lambda = float(config.get('cache_lambda', 0.0))
        self.cache_theta = float(config.get('cache_theta', 20.0))
        self.logit_temperature = float(config.get('logit_temperature', 1.0))
        if not 0.0 <= self.cache_lambda < 1.0 or self.cache_theta <= 0.0:
            raise ValueError('cache_lambda must be in [0, 1) and cache_theta must be positive.')
        if self.logit_temperature <= 0.0:
            raise ValueError('logit_temperature must be positive.')
        width = config['width']
        rmsnorm = bool(config.get('rmsnorm', True))
        swiglu = bool(config.get('swiglu', True))
        self.moe_experts = int(config.get('moe_experts', 1))
        if self.moe_experts < 1 or (self.moe_experts > 1 and not swiglu):
            raise ValueError('moe_experts must be positive and requires swiglu=true.')
        dropout = float(config.get('dropout', 0.0))
        # Legacy configs use one dropout rate for all three locations.
        attention_dropout = float(config.get('attention_dropout', dropout))
        residual_dropout = float(config.get('residual_dropout', dropout))
        self.embedding_dropout = float(config.get('embedding_dropout', dropout))
        for name, probability in (
            ('dropout', dropout), ('attention_dropout', attention_dropout),
            ('residual_dropout', residual_dropout), ('embedding_dropout', self.embedding_dropout)
        ):
            if not 0.0 <= probability < 1.0:
                raise ValueError(f'{name} must be in [0, 1).')
        self.token = nn.Embedding(config['vocab'], width)
        self.pos = None if self.use_rope else nn.Embedding(self.context, width)
        head_dim = width // config['heads']
        cos, sin = rope_cache(self.context, head_dim)
        # These are deterministic, so checkpoints do not need to store them.
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)
        self.blocks = nn.ModuleList(
            [Block(width, config['heads'], rmsnorm, swiglu, dropout,
                   config.get('ffn_multiplier'), self.moe_experts,
                   attention_dropout=attention_dropout, residual_dropout=residual_dropout)
             for _ in range(config['depth'])]
        )
        self.norm = RMSNorm(width) if rmsnorm else nn.LayerNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=bool(config.get('head_bias', False)))
        self.mtp_weight = float(config.get('mtp_weight', 0.0))
        self.mtp_heads = nn.ModuleList([
            nn.Linear(width, config['vocab'], bias=False)
            for _ in range(int(config.get('mtp_heads', 0)))
        ])
        if self.mtp_weight < 0.0:
            raise ValueError('mtp_weight cannot be negative.')
        self.apply(self.initialize)
        if config.get('scale_residual_init', True):
            residual_std = .02 / (2 * config['depth']) ** .5
            for block in self.blocks:
                nn.init.normal_(block.proj.weight, std=residual_std)
                if isinstance(block.mlp, TokenRoutedMoE):
                    for expert in block.mlp.experts:
                        nn.init.normal_(expert.down.weight, std=residual_std)
                else:
                    output = block.mlp.down if isinstance(block.mlp, SwiGLU) else block.mlp[-1]
                    nn.init.normal_(output.weight, std=residual_std)
        if config.get('tie_embeddings', True):
            self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        x = F.dropout(self.token(ids), self.embedding_dropout, self.training)
        if self.pos is not None:
            x = x + self.pos(torch.arange(ids.shape[1], device=ids.device))
        rope = (self.rope_cos, self.rope_sin) if self.use_rope else None
        routes = ids.remainder(self.moe_experts) if self.moe_experts > 1 else None
        for block in self.blocks:
            x = block(x, rope, routes)
        return self.norm(x)

    def forward(self, ids):
        return self.head(self.features(ids))

    def training_loss(self, batch):
        """Next-token loss plus training-only multi-token prediction heads."""
        features = self.features(batch[:, :-1])
        main_loss = F.cross_entropy(
            self.head(features).flatten(0, 1).float(), batch[:, 1:].flatten()
        )
        if not self.mtp_heads or self.mtp_weight == 0.0:
            return main_loss
        auxiliary = []
        for index, head in enumerate(self.mtp_heads):
            horizon = index + 2
            usable = features.shape[1] - horizon + 1
            auxiliary.append(F.cross_entropy(
                head(features[:, :usable]).flatten(0, 1).float(),
                batch[:, horizon:].flatten(),
            ))
        return main_loss + self.mtp_weight * torch.stack(auxiliary).mean()

    def predict_log_probs(self, ids):
        features = self.features(ids)
        base_logp = F.log_softmax(
            self.head(features).float() / self.logit_temperature, dim=-1
        )
        if self.cache_lambda <= 0.0 or ids.shape[1] < 2:
            return base_logp

        # A query at t may match keys i<t and copy the already observed outcome
        # ids[i+1]. No future token, cross-example state, or cross-window state is used.
        normalized = F.normalize(features.float(), dim=-1)
        similarity = normalized @ normalized.transpose(-1, -2)
        length = ids.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=ids.device).tril(-1)
        similarity = (self.cache_theta * similarity).masked_fill(~causal, float('-inf'))
        # Avoid an all-masked softmax for t=0; that row is replaced by base_logp.
        similarity[:, 0, 0] = 0.0
        weights = similarity.softmax(dim=-1)
        cache_p = torch.zeros_like(base_logp)
        outcomes = torch.cat((ids[:, 1:], ids[:, -1:]), dim=1)
        cache_p.scatter_add_(2, outcomes[:, None, :].expand(-1, length, -1), weights)
        mixed = torch.logaddexp(
            base_logp + math.log1p(-self.cache_lambda),
            cache_p.clamp_min(1e-30).log() + math.log(self.cache_lambda),
        )
        mixed[:, 0] = base_logp[:, 0]
        return mixed


def build_model(config):
    return StudentGPT(config)

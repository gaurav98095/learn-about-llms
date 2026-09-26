"""A complete, readable GPT-2 architecture implemented directly in PyTorch."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class GPT2Config:
    """GPT-2 configuration. Defaults are GPT-2 small (124M parameters)."""

    vocab_size: int = 50_257
    context_length: int = 1_024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.vocab_size <= 0 or self.context_length <= 0:
            raise ValueError("vocab_size and context_length must be positive")
        if self.n_layer <= 0 or self.n_head <= 0 or self.n_embd <= 0:
            raise ValueError("n_layer, n_head, and n_embd must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """GPT-2 masked multi-head self-attention with one fused QKV projection."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.context_length, config.context_length)).view(
                1, 1, config.context_length, config.context_length
            ),
            persistent=False,
        )
        self.c_proj._gpt2_residual_projection = True

    def forward(self, x: Tensor) -> Tensor:
        batch, time, channels = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, time, self.n_head, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / self.head_dim**0.5
        mask = self.bias[:, :, :time, :time].bool()
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = self.attn_dropout(F.softmax(scores, dim=-1))

        y = weights @ v
        y = y.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.c_proj._gpt2_residual_projection = True

    def forward(self, x: Tensor) -> Tensor:
        x = F.gelu(self.c_fc(x), approximate="tanh")
        return self.dropout(self.c_proj(x))


class TransformerBlock(nn.Module):
    """One GPT-2 pre-normalized residual block."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    """Decoder-only GPT-2 language model."""

    def __init__(self, config: GPT2Config | None = None) -> None:
        super().__init__()
        self.config = config or GPT2Config()
        config = self.config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.context_length, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f = LayerNorm(config.n_embd, config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            std = 0.02
            if isinstance(module, nn.Linear) and getattr(
                module, "_gpt2_residual_projection", False
            ):
                std /= (2 * self.config.n_layer) ** 0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, token_ids: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, time)")
        _, time = token_ids.shape
        if time > self.config.context_length:
            raise ValueError("sequence exceeds the model context length")

        positions = torch.arange(time, device=token_ids.device)
        x = self.drop(self.wte(token_ids) + self.wpe(positions))
        for block in self.h:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            if targets.shape != token_ids.shape:
                raise ValueError("targets must have the same shape as token_ids")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        token_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.config.context_length :]
            logits, _ = self(context)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)
        return token_ids


if __name__ == "__main__":
    model = GPT2(GPT2Config(n_layer=2, n_head=4, n_embd=128, context_length=64))
    ids = torch.randint(0, model.config.vocab_size, (1, 8))
    logits, loss = model(ids, ids)
    print(logits.shape, loss.item())

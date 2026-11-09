"""PyTorch decoder-only Transformer for Simple AI.

This backend is intended for cloud training. The educational custom-autograd
implementation remains separate and does not require PyTorch.
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    context_size: int = 256
    embedding_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive.")
        if self.context_size < 1:
            raise ValueError("context_size must be positive.")
        if self.embedding_dim < 1:
            raise ValueError("embedding_dim must be positive.")
        if self.num_heads < 1:
            raise ValueError("num_heads must be positive.")
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention using PyTorch's optimized SDPA."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        self.dropout = config.dropout

        self.qkv = nn.Linear(config.embedding_dim, 3 * config.embedding_dim)
        self.projection = nn.Linear(
            config.embedding_dim, config.embedding_dim
        )
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, channels = x.shape

        qkv = self.qkv(x).view(
            batch, sequence, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch, sequence, channels
        )

        return self.residual_dropout(self.projection(attended))


class TransformerBlock(nn.Module):
    """Pre-normalized Transformer block."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        width = config.embedding_dim

        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.attention = CausalSelfAttention(config)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class GPTLanguageModel(nn.Module):
    """Decoder-only language model trained to predict the next token."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size, config.embedding_dim
        )
        self.position_embedding = nn.Embedding(
            config.context_size, config.embedding_dim
        )
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(
            config.embedding_dim, config.vocab_size, bias=False
        )

        # Share input-token and output-token weights (weight tying).
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")

        _, sequence = input_ids.shape
        if sequence < 1 or sequence > self.config.context_size:
            raise ValueError("Input sequence is outside the supported context.")

        positions = torch.arange(sequence, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        logits = self.lm_head(self.final_norm(x))
        loss = None

        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids.")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.size(1) < 1:
            raise ValueError("input_ids must be a non-empty 2D tensor.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative.")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive.")

        was_training = self.training
        self.eval()

        try:
            for _ in range(max_new_tokens):
                context = input_ids[:, -self.config.context_size :]
                logits, _ = self(context)
                logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    values, _ = torch.topk(logits, k)
                    cutoff = values[:, [-1]]
                    logits = logits.masked_fill(
                        logits < cutoff, float("-inf")
                    )

                probabilities = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
                input_ids = torch.cat((input_ids, next_token), dim=1)

                if eos_id is not None and torch.all(next_token == eos_id):
                    break
        finally:
            self.train(was_training)

        return input_ids

    def get_config(self) -> dict[str, int | float]:
        return asdict(self.config)

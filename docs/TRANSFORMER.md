# Tiny causal Transformer

Simple AI includes an educational decoder-only Transformer block built on its
own 2-D Tensor/autograd engine.

## Architecture

- Token embedding plus learned positional embedding
- Single-head causal self-attention (query, key, value projections)
- Residual connection
- Two-layer ReLU feed-forward network and residual connection
- Vocabulary projection from the final context position

The causal mask prevents a position from attending to later positions. The
implementation uses a single attention head and one block; it has no layer
normalization, dropout, KV cache, or optimized matrix kernels.

## Train

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model \
  --tokenizer bpe --architecture transformer --vocab-size 128 \
  --context-size 16 --embedding-dim 8 --epochs 5 --batch-size 8
```

The model checkpoint and tokenizer are written together to
`artifacts/indonesian_model/`. Start chat with the same pair:

```bash
PYTHONPATH=python python -m simple_ai.chat
```

## Performance and limitations

This implementation loops over examples and uses pure-Python matrix
operations. It is for correctness and learning, not speed. Start with a small
context, embedding dimension, and dataset on Android. Benchmark wall time and
peak memory before increasing them. A Transformer architecture alone does not
create language knowledge; training data and independent evaluation remain
essential.

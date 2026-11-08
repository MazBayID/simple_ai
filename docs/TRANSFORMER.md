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
normalization, dropout, KV cache, or optimized matrix kernels. Transformer
training defaults to Adam at learning rate 0.001; the MLP baseline defaults
to SGD at 0.03. Both can be selected explicitly with `--optimizer` and
`--learning-rate`.

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

## Independent validation

For a more meaningful check, provide a separate corpus that was not used to
fit the tokenizer or train the model:

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model \
  --input data/train.txt --validation-input data/validation.txt \
  --tokenizer bpe --architecture transformer --context-size 16 \
  --embedding-dim 8 --epochs 5 --batch-size 8
```

The report labels this as an external validation source. Use a third, untouched
test corpus for final model comparisons; do not repeatedly tune against it.


## Benchmark on Termux

After training, measure checkpoint loading and generation:

```bash
PYTHONPATH=python python -m simple_ai.experiments.benchmark_model \
  --generation-tokens 32 --prompt "Halo, apa kabar?"
```

The report includes load time, generated tokens per second, checkpoint size,
and Python-traced peak allocation. The memory figure is not total Android
process RSS; use Android/Termux system tools for process-level RAM profiling.

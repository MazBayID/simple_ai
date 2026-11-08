# Token Embedding Layer

The embedding milestone adds a trainable lookup table to the pure-Python
neural engine.

## Data flow

For a batch of fixed-length contexts, the layer maps each token ID to a
trainable vector and concatenates the vectors in their original position
order.

For example, with batch size 2, context length 3, and embedding dimension 4:

- Input token IDs: shape `(2, 3)`
- Embedding table: shape `(vocabulary_size, 4)`
- Output features: shape `(2, 12)`

The flattened representation is a deliberate compatibility bridge because
the current Tensor engine supports 2-D matrices, not 3-D tensors. Position
order is retained, so a later linear projection can learn different weights
for each context position.

## Gradients

During backpropagation, the layer routes each output gradient back to the
corresponding row in the embedding table. If the same token occurs more than
once, its gradients are accumulated. This is the same parameter-sharing
principle used by conventional embedding layers.

## Run tests

```bash
PYTHONPATH=python python -m pytest -q tests/test_embedding.py
```

## Limitations

- Character IDs are used by the current tokenizer; no subword tokenizer yet.
- Inputs are fixed-length and must form a rectangular batch.
- The layer returns flattened 2-D features, not a 3-D sequence tensor.
- Padding masks, sparse gradient updates, and optimized lookup kernels are
  not implemented.
- This layer alone does not constitute a language model.


The training objective now also includes a numerically stable cross-entropy loss
for raw vocabulary logits. See [CROSS_ENTROPY.md](CROSS_ENTROPY.md).

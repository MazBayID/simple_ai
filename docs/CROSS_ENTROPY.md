# Cross-Entropy Loss

This milestone adds a classification loss for next-token prediction.

## Inputs and calculation

The loss receives a 2-D logits tensor with shape
`(batch_size, vocabulary_size)` and one integer target token ID per row.
Logits are raw model scores; they do not need to be converted to probabilities
before calling the loss.

For each row, the implementation calculates the stable log-sum-exp form:

```text
loss = log(sum(exp(logit))) - target_logit
```

It subtracts the row maximum before exponentiation to avoid overflow. The
batch loss is the mean across examples.

## Backpropagation

The derivative with respect to each logit is:

```text
(softmax(logits) - one_hot(target)) / batch_size
```

The loss is connected to the project's Tensor graph, so gradients flow back
into a future vocabulary projection layer.

## Run tests

```bash
PYTHONPATH=python python -m pytest -q tests/test_losses.py
```

## Limitations

This loss does not create the logits or the language model. A trainable output
projection and a complete next-token training loop are still required.

# Neural Network and Training Engine

This milestone adds a small trainable neural-network foundation on top of the
pure-Python Tensor engine.

## Components

- `simple_ai.nn.Linear`: fully connected matrix layer (bias is not included yet).
- `simple_ai.nn.ReLU`: ReLU activation wrapper.
- `simple_ai.losses.MeanSquaredError`: mean squared error with automatic gradients.
- `simple_ai.optim.SGD`: stochastic gradient descent over leaf tensors.
- `simple_ai.training.train_step`: one forward, loss, backward, and update step.
- `simple_ai.training.fit`: repeated full-batch training with optional callback.

All operations use the project's own Tensor implementation and Python standard
library. No NumPy, PyTorch, TensorFlow, or network access is required at runtime.

## Run a real training experiment

From the repository root:

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_regression
```

The experiment trains a one-weight linear model on the relation `y = 2x`,
prints initial and final loss, and writes
`artifacts/regression_checkpoint.json`.

GitHub Actions can run the same experiment and upload the checkpoint as a
workflow artifact. This is a neural-network training smoke test, not a language
model and not evidence of conversational intelligence.

## Current limitations

- Linear layer currently has no bias.
- Training is full-batch and CPU-only.
- Only the operations implemented by the Tensor engine are differentiable.
- The regression task is deliberately simple and deterministic.
- There is no tokenizer, Transformer, language-model dataset, or chat inference
  in this milestone.
- Resource targets must be measured separately; passing this experiment does
  not establish the project's RAM/CPU target.

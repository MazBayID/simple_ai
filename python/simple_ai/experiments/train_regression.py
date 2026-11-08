"""Train a tiny neural network on a deterministic linear-regression task.

This is a training-engine smoke test, not language-model training.
Run with:
    PYTHONPATH=python python -m simple_ai.experiments.train_regression
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from simple_ai.core import Tensor
from simple_ai.nn import Linear
from simple_ai.optim import SGD
from simple_ai.training import fit


class RegressionModel:
    def __init__(self, seed: int = 7) -> None:
        self.output = Linear(1, 1, seed=seed)

    def __call__(self, inputs: Tensor) -> Tensor:
        return self.output(inputs)

    def parameters(self) -> list[Tensor]:
        return self.output.parameters()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=Path("artifacts/regression_checkpoint.json"))
    args = parser.parse_args()

    inputs = Tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]], requires_grad=False)
    targets = Tensor([[-4.0], [-2.0], [0.0], [2.0], [4.0]], requires_grad=False)

    model = RegressionModel()
    optimizer = SGD(model.parameters(), learning_rate=args.learning_rate)
    history = fit(model, inputs, targets, optimizer, epochs=args.epochs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "simple-ai-linear-regression-v1",
        "task": "y = 2x",
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "final_loss": history[-1],
        "weights": model.output.weights.data,
    }
    args.output.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

    print(f"Initial loss: {history[0]:.8f}")
    print(f"Final loss:   {history[-1]:.8f}")
    print(f"Learned weight: {model.output.weights.data[0][0]:.6f}")
    print(f"Checkpoint: {args.output}")
    if history[-1] >= history[0]:
        raise SystemExit("Training did not reduce loss; experiment failed.")


if __name__ == "__main__":
    main()

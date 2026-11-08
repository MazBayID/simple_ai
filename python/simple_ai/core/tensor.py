"""A tiny 2-D tensor with reverse-mode automatic differentiation.

This module uses Python lists and the standard library only. It is an
educational engine, not a high-performance numerical library.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

Matrix = list[list[float]]


def _matrix(data: float | int | Sequence[Sequence[float]]) -> Matrix:
    if isinstance(data, (int, float)):
        return [[float(data)]]
    rows = [[float(value) for value in row] for row in data]
    if not rows or not rows[0]:
        raise ValueError("Tensor data must contain at least one value.")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("Tensor data must be rectangular.")
    return rows


def _zeros(shape: tuple[int, int]) -> Matrix:
    return [[0.0 for _ in range(shape[1])] for _ in range(shape[0])]


def _transpose(data: Matrix) -> Matrix:
    return [list(column) for column in zip(*data)]


class Tensor:
    """A rectangular 2-D tensor supporting a small set of differentiable ops.

    Supported operations: addition, element-wise multiplication, matrix
    multiplication, transpose, ReLU, sum, mean, and scalar arithmetic.

    Call backward() on a scalar result to accumulate gradients in leaf tensors.
    """

    def __init__(
        self,
        data: float | int | Sequence[Sequence[float]],
        *,
        requires_grad: bool = True,
        _parents: tuple["Tensor", ...] = (),
        _backward: Callable[[], None] | None = None,
    ) -> None:
        self.data = _matrix(data)
        self.shape = (len(self.data), len(self.data[0]))
        self.requires_grad = requires_grad
        self.grad = _zeros(self.shape)
        self._parents = _parents
        self._backward = _backward or (lambda: None)

    def _add_grad(self, values: Matrix) -> None:
        if not self.requires_grad:
            return
        for row in range(self.shape[0]):
            for column in range(self.shape[1]):
                self.grad[row][column] += values[row][column]

    @staticmethod
    def _coerce(value: "Tensor | float | int") -> "Tensor":
        if isinstance(value, Tensor):
            return value
        return Tensor(value, requires_grad=False)

    def __add__(self, other: "Tensor | float | int") -> "Tensor":
        if isinstance(other, (int, float)):
            other = Tensor(
                [[float(other) for _ in range(self.shape[1])] for _ in range(self.shape[0])],
                requires_grad=False,
            )
        else:
            other = self._coerce(other)
        if self.shape != other.shape:
            raise ValueError(f"Addition requires equal shapes: {self.shape} and {other.shape}.")
        result = Tensor(
            [[a + b for a, b in zip(ar, br)] for ar, br in zip(self.data, other.data)],
            requires_grad=self.requires_grad or other.requires_grad,
            _parents=(self, other),
        )

        def backward() -> None:
            self._add_grad(result.grad)
            other._add_grad(result.grad)

        result._backward = backward
        return result

    __radd__ = __add__

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __sub__(self, other: "Tensor | float | int") -> "Tensor":
        if isinstance(other, (int, float)):
            return self + (-float(other))
        return self + (-other)

    def __rsub__(self, other: "Tensor | float | int") -> "Tensor":
        return (-self) + other

    def __mul__(self, other: "Tensor | float | int") -> "Tensor":
        if isinstance(other, (int, float)):
            other = Tensor(
                [[float(other) for _ in range(self.shape[1])] for _ in range(self.shape[0])],
                requires_grad=False,
            )
        else:
            other = self._coerce(other)
        if self.shape != other.shape:
            raise ValueError(
                f"Element-wise multiplication requires equal shapes: {self.shape} and {other.shape}."
            )
        result = Tensor(
            [[a * b for a, b in zip(ar, br)] for ar, br in zip(self.data, other.data)],
            requires_grad=self.requires_grad or other.requires_grad,
            _parents=(self, other),
        )

        def backward() -> None:
            self._add_grad([
                [g * b for g, b in zip(gr, br)]
                for gr, br in zip(result.grad, other.data)
            ])
            other._add_grad([
                [g * a for g, a in zip(gr, ar)]
                for gr, ar in zip(result.grad, self.data)
            ])

        result._backward = backward
        return result

    __rmul__ = __mul__

    def __matmul__(self, other: "Tensor") -> "Tensor":
        if not isinstance(other, Tensor):
            raise TypeError("Matrix multiplication requires another Tensor.")
        if self.shape[1] != other.shape[0]:
            raise ValueError(f"Cannot multiply shapes {self.shape} and {other.shape}.")
        rows, inner, columns = self.shape[0], self.shape[1], other.shape[1]
        values = [
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(inner))
                for j in range(columns)
            ]
            for i in range(rows)
        ]
        result = Tensor(
            values,
            requires_grad=self.requires_grad or other.requires_grad,
            _parents=(self, other),
        )

        def backward() -> None:
            if self.requires_grad:
                right_t = _transpose(other.data)
                self._add_grad([
                    [
                        sum(result.grad[i][k] * right_t[k][j] for k in range(columns))
                        for j in range(inner)
                    ]
                    for i in range(rows)
                ])
            if other.requires_grad:
                left_t = _transpose(self.data)
                other._add_grad([
                    [
                        sum(left_t[i][k] * result.grad[k][j] for k in range(rows))
                        for j in range(columns)
                    ]
                    for i in range(inner)
                ])

        result._backward = backward
        return result

    def transpose(self) -> "Tensor":
        result = Tensor(
            _transpose(self.data),
            requires_grad=self.requires_grad,
            _parents=(self,),
        )

        def backward() -> None:
            self._add_grad(_transpose(result.grad))

        result._backward = backward
        return result

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def relu(self) -> "Tensor":
        result = Tensor(
            [[max(0.0, value) for value in row] for row in self.data],
            requires_grad=self.requires_grad,
            _parents=(self,),
        )

        def backward() -> None:
            self._add_grad([
                [
                    grad if value > 0.0 else 0.0
                    for grad, value in zip(grad_row, data_row)
                ]
                for grad_row, data_row in zip(result.grad, self.data)
            ])

        result._backward = backward
        return result

    def sum(self) -> "Tensor":
        total = sum(sum(row) for row in self.data)
        result = Tensor(total, requires_grad=self.requires_grad, _parents=(self,))

        def backward() -> None:
            self._add_grad([
                [result.grad[0][0] for _ in range(self.shape[1])]
                for _ in range(self.shape[0])
            ])

        result._backward = backward
        return result

    def mean(self) -> "Tensor":
        return self.sum() * (1.0 / (self.shape[0] * self.shape[1]))

    def backward(self) -> None:
        """Backpropagate from a scalar tensor and accumulate leaf gradients."""
        if self.shape != (1, 1):
            raise ValueError("backward() requires a scalar tensor; call sum() or mean() first.")
        if not self.requires_grad:
            raise RuntimeError("Cannot backpropagate from a tensor that does not require gradients.")

        ordered: list[Tensor] = []
        visited: set[int] = set()

        def visit(node: Tensor) -> None:
            identity = id(node)
            if identity in visited:
                return
            visited.add(identity)
            for parent in node._parents:
                visit(parent)
            ordered.append(node)

        visit(self)
        self.grad[0][0] += 1.0
        for node in reversed(ordered):
            node._backward()

    def item(self) -> float:
        if self.shape != (1, 1):
            raise ValueError(f"item() requires a scalar tensor, got shape {self.shape}.")
        return self.data[0][0]

    def __repr__(self) -> str:
        return (
            f"Tensor(data={self.data!r}, grad={self.grad!r}, "
            f"requires_grad={self.requires_grad})"
        )

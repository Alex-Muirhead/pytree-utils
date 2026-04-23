"""Elementwise mathematical operations mixin for ArrayTree."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def _binop(op: Any, left: Any, right: Any) -> Any:
    """Apply ``op(left_leaf, right_leaf)`` elementwise across a tree."""
    return jax.tree.map(op, left, jax.tree.broadcast(right, left))


def _rbinop(op: Any, left: Any, right: Any) -> Any:
    """Apply ``op(right_leaf, left_leaf)`` elementwise — for reflected ops."""
    return jax.tree.map(op, jax.tree.broadcast(right, left), left)


class _ArrayTreeOps:
    """Mixin that provides elementwise mathematical operations for ArrayTree.

    Every operation is applied leaf-by-leaf via ``jax.tree.map``.  The
    right-hand operand is first broadcast to the tree's structure (so a plain
    scalar or JAX array is accepted alongside a same-structure ArrayTree).
    """

    # ------------------------------------------------------------------
    # Unary
    # ------------------------------------------------------------------

    def __neg__(self) -> Any:
        return jax.tree.map(jnp.negative, self)

    def __pos__(self) -> Any:
        return jax.tree.map(jnp.positive, self)

    def __abs__(self) -> Any:
        return jax.tree.map(jnp.abs, self)

    # ------------------------------------------------------------------
    # Binary arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: Any) -> Any:
        return _binop(jnp.add, self, other)

    def __radd__(self, other: Any) -> Any:
        return _binop(jnp.add, self, other)  # addition is commutative

    def __sub__(self, other: Any) -> Any:
        return _binop(jnp.subtract, self, other)

    def __rsub__(self, other: Any) -> Any:
        return _rbinop(jnp.subtract, self, other)

    def __mul__(self, other: Any) -> Any:
        return _binop(jnp.multiply, self, other)

    def __rmul__(self, other: Any) -> Any:
        return _binop(jnp.multiply, self, other)  # multiplication is commutative

    def __truediv__(self, other: Any) -> Any:
        return _binop(jnp.divide, self, other)

    def __rtruediv__(self, other: Any) -> Any:
        return _rbinop(jnp.divide, self, other)

    def __floordiv__(self, other: Any) -> Any:
        return _binop(jnp.floor_divide, self, other)

    def __rfloordiv__(self, other: Any) -> Any:
        return _rbinop(jnp.floor_divide, self, other)

    def __mod__(self, other: Any) -> Any:
        return _binop(jnp.mod, self, other)

    def __rmod__(self, other: Any) -> Any:
        return _rbinop(jnp.mod, self, other)

    def __pow__(self, other: Any) -> Any:
        return _binop(jnp.power, self, other)

    def __rpow__(self, other: Any) -> Any:
        return _rbinop(jnp.power, self, other)

    # ------------------------------------------------------------------
    # Comparison  (elementwise — returns a same-structure tree of bools)
    # ------------------------------------------------------------------

    def __lt__(self, other: Any) -> Any:
        return _binop(jnp.less, self, other)

    def __le__(self, other: Any) -> Any:
        return _binop(jnp.less_equal, self, other)

    def __gt__(self, other: Any) -> Any:
        return _binop(jnp.greater, self, other)

    def __ge__(self, other: Any) -> Any:
        return _binop(jnp.greater_equal, self, other)

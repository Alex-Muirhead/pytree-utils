"""Elementwise mathematical operations mixin for ArrayTree."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

type UnaryFunc[T] = Callable[[T], T]
type BinaryFunc[T] = Callable[[T, T], T]


def unary_op(op: UnaryFunc[ArrayLike], name: str) -> UnaryFunc[Any]:
    """Create a unary operator method."""

    def wrapper(this_tree):  # noqa: ANN001, ANN202
        return jax.tree.map(op, this_tree)

    wrapper.__name__ = f"__{name}__"
    return wrapper


def binary_op(op: BinaryFunc[ArrayLike], name: str) -> BinaryFunc:
    """Create a binary operator method."""

    def wrapper(this_tree, other):  # noqa: ANN001, ANN202
        other_tree = jax.tree.broadcast(other, this_tree)
        return jax.tree.map(op, this_tree, other_tree)

    wrapper.__name__ = f"__{name}__"
    return wrapper


def reflected_binary_op(op: BinaryFunc[ArrayLike], name: str) -> BinaryFunc:
    """Create a reflected binary operator method."""

    def wrapper(this_tree, other):  # noqa: ANN001, ANN202
        other_tree = jax.tree.broadcast(other, this_tree)
        return jax.tree.map(op, other_tree, this_tree)

    wrapper.__name__ = f"__r{name}__"
    return wrapper


def numerical_binary_op(op: BinaryFunc[ArrayLike], name: str) -> tuple[BinaryFunc, BinaryFunc]:
    """Create a pair of binary operator methods."""
    return binary_op(op, name), reflected_binary_op(op, name)


class ArrayTreeOps:
    """Elementwise arithmetic and comparison operators for ArrayTree.

    Each operator maps over the leaves with ``jax.tree.map``, broadcasting the
    right-hand operand to the tree first so a scalar, a JAX array, or another
    same-structure ArrayTree are all accepted.
    """

    # Unary

    __neg__ = unary_op(jnp.negative, "neg")
    __pos__ = unary_op(jnp.negative, "pos")
    __abs__ = unary_op(jnp.negative, "abs")

    # Binary

    __add__, __radd__ = numerical_binary_op(jnp.add, "add")
    __sub__, __rsub__ = numerical_binary_op(jnp.subtract, "sub")
    __mul__, __rmul__ = numerical_binary_op(jnp.multiply, "mul")
    __pow__, __rpow__ = numerical_binary_op(jnp.power, "pow")
    __mod__, __rmod__ = numerical_binary_op(jnp.mod, "mod")

    __truediv__, __rtruediv__ = numerical_binary_op(jnp.true_divide, "truediv")
    __floordiv__, __rfloordiv__ = numerical_binary_op(jnp.floor_divide, "floordiv")

    # Ordering

    __lt__ = binary_op(jnp.less, "lt")
    __le__ = binary_op(jnp.less_equal, "le")
    __gt__ = binary_op(jnp.greater, "gt")
    __ge__ = binary_op(jnp.greater_equal, "ge")

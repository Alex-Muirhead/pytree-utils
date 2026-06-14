"""Index helpers (.at[..] scatter/gather operations) for ArrayTree."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import jax
import numpy as np
from jax.typing import ArrayLike

if TYPE_CHECKING:
    from pytree_utils.array_tree import ArrayTree


type AnyInt = int | np.integer
type StaticIndex = AnyInt | slice | type[Ellipsis]  # ty:ignore[invalid-type-form]
type Index = StaticIndex | None | Sequence[AnyInt] | jax.Array | np.ndarray


@dataclass(frozen=True)
class IndexHelperRef[T: ArrayTree]:
    """Helper object to call indexed update functions for an (advanced) index."""

    node: T
    idx: tuple

    def _scatter(self, op: str, values: T | ArrayLike, **kwargs: object) -> T:
        """Broadcast *values* to the focus, apply ``arr.at[idx].<op>``."""
        values_tree = jax.tree.broadcast(values, self.node)
        return jax.tree.map(
            lambda array, value: getattr(array.at[..., *self.idx], op)(value, **kwargs),
            self.node,
            values_tree,
        )

    def get(self, **kwargs: object) -> T:
        """Equivalent to ``leaf[idx]`` for indexed leaves."""
        return jax.tree.map(lambda arr: arr.at[..., *self.idx].get(**kwargs), self.node)

    def set(self, values: T | ArrayLike, **kwargs: object) -> T:
        """Pure equivalent to ``leaf[idx] = value`` for indexed leaves."""
        return self._scatter("set", values, **kwargs)

    def add(self, values: T | ArrayLike, **kwargs: object) -> T:
        """Pure equivalent to ``leaf[idx] += value`` for indexed leaves."""
        return self._scatter("add", values, **kwargs)

    def subtract(self, values: T | ArrayLike, **kwargs: object) -> T:
        """Pure equivalent to ``leaf[idx] -= value`` for indexed leaves."""
        return self._scatter("subtract", values, **kwargs)

    def multiply(self, values: T | ArrayLike, **kwargs: object) -> T:
        """Pure equivalent to ``leaf[idx] *= value`` for indexed leaves."""
        return self._scatter("multiply", values, **kwargs)

    def divide(self, values: T | ArrayLike, **kwargs: object) -> T:
        """Pure equivalent to ``leaf[idx] /= value`` for indexed leaves."""
        return self._scatter("divide", values, **kwargs)

    def power(self, values: T | ArrayLike, **kwargs: object) -> T:
        """Pure equivalent to ``leaf[idx] **= value`` for indexed leaves."""
        return self._scatter("power", values, **kwargs)

    def min(self, values: T | ArrayLike, **kwargs: object) -> T:
        """Pure equivalent to ``leaf[idx] = minimum(leaf[idx], value)`` for indexed leaves."""
        return self._scatter("min", values, **kwargs)

    def max(self, values: T | ArrayLike, **kwargs: object) -> T:
        """Pure equivalent to ``leaf[idx] = maximum(leaf[idx], value)`` for indexed leaves."""
        return self._scatter("max", values, **kwargs)

    def apply(self, func: Callable, **kwargs: object) -> T:
        """Pure equivalent to ``leaf[idx] = func(leaf[idx], **kwargs)`` for indexed leaves."""
        return jax.tree.map(lambda arr: arr.at[..., *self.idx].apply(func, **kwargs), self.node)


@dataclass(frozen=True)
class IndexHelper[T: ArrayTree]:
    """Helper property for index update functionality."""

    node: T

    def __getitem__(self, idx: Index | Sequence[Index]) -> IndexHelperRef:
        normed_idx = cast(tuple[Index, ...], jax.numpy.index_exp[idx])
        ndim = count_index_dims(normed_idx)

        if ndim > self.node.ndim:
            raise IndexError(
                f"Too many indicies: node {self.node} is "
                f"{self.node.ndim}-dimensional but {ndim} were indexed"
            )

        return IndexHelperRef(node=self.node, idx=normed_idx)


def count_index_dims(idx: tuple) -> int:
    """Count how many existing dimensions an index tuple addresses."""
    return sum(i is not None for i in idx)

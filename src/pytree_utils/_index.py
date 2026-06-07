"""Index helpers (.at[..] scatter/gather operations) for ArrayTree."""

import dataclasses as dc
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax

from pytree_utils._spec import count_index_dims

if TYPE_CHECKING:
    from pytree_utils.array_tree import ArrayTree


@dc.dataclass(frozen=True)
class IndexedHelper[T: ArrayTree]:
    """Returned by ``ArrayTree.at[idx]``; mirrors JAX's scatter/gather API."""

    node: T
    idx: tuple

    def _scatter(self, op: str, values: Any, **kwargs: Any) -> T:
        """Broadcast *values* to the tree and apply ``arr.at[idx].<op>``."""
        idx = self.idx
        values_tree = jax.tree.broadcast(values, self.node)
        return jax.tree.map(
            lambda arr, v: getattr(arr.at[idx], op)(v, **kwargs),
            self.node,
            values_tree,
        )

    def get(self, **kwargs: Any) -> T:
        """Return a new node with ``idx`` applied to every leaf.

        Each node's ``_shape`` leaf is indexed alongside the data, so the
        reported shapes stay correct without any extra bookkeeping.
        """
        idx = self.idx
        return jax.tree.map(lambda arr: arr.at[idx].get(**kwargs), self.node)

    def set(self, values: Any, **kwargs: Any) -> T:
        """Return a copy with indexed leaves replaced by *values*."""
        return self._scatter("set", values, **kwargs)

    def add(self, values: Any, **kwargs: Any) -> T:
        """Return a copy with *values* added to the indexed leaves."""
        return self._scatter("add", values, **kwargs)

    def mul(self, values: Any, **kwargs: Any) -> T:
        """Return a copy with indexed leaves multiplied by *values*."""
        return self._scatter("mul", values, **kwargs)

    def min(self, values: Any, **kwargs: Any) -> T:
        """Return a copy with indexed leaves replaced by min(leaf, value)."""
        return self._scatter("min", values, **kwargs)

    def max(self, values: Any, **kwargs: Any) -> T:
        """Return a copy with indexed leaves replaced by max(leaf, value)."""
        return self._scatter("max", values, **kwargs)

    def apply(self, func: Callable, **kwargs: Any) -> T:
        """Return a copy with *func* applied to each indexed leaf slice."""
        idx = self.idx
        return jax.tree.map(lambda arr: arr.at[idx].apply(func, **kwargs), self.node)


@dc.dataclass(frozen=True)
class IndexHelper[T: ArrayTree]:
    """Returned by ``ArrayTree.at``; validates the index and captures it."""

    node: T

    def __getitem__(self, idx: Any) -> IndexedHelper[T]:
        if not isinstance(idx, tuple):
            idx = (idx,)

        shape = self.node.shape
        n = count_index_dims(idx)
        if n > len(shape):
            raise IndexError(
                f"{type(self.node).__name__} has a {len(shape)}-dimensional "
                f"prefix {shape}; cannot index with {n} dimension(s)"
            )

        return IndexedHelper(self.node, idx)

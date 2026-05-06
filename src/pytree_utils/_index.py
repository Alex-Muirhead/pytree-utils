"""Index helpers (.at[..] scatter/gather operations) for ArrayTree."""

import dataclasses as dc
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax

from pytree_utils._spec import _count_index_dims

if TYPE_CHECKING:
    from pytree_utils.array_tree import ArrayTree


@dc.dataclass(frozen=True)
class _IndexedHelper[T: ArrayTree]:
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
        """Return a new node with ``idx`` applied to all leaf arrays."""
        full_prefix = self.node._prefix + self.node.shape
        remaining = full_prefix[_count_index_dims(self.idx) :]
        return self.node._reindex(self.idx, new_prefix=(), new_shape=remaining, get_kw=kwargs or None)

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
class _IndexHelper[T: ArrayTree]:
    """Returned by ``ArrayTree.at``; validates the index and captures it."""

    node: T

    def __getitem__(self, idx: Any) -> _IndexedHelper[T]:
        if not isinstance(idx, tuple):
            idx = (idx,)

        full_prefix = self.node._prefix + self.node.shape
        n = _count_index_dims(idx)

        if n > len(full_prefix):
            raise IndexError(
                f"{type(self.node).__name__} has a {len(full_prefix)}-dimensional "
                f"prefix {full_prefix}; cannot index with {n} dimension(s)"
            )

        return _IndexedHelper(self.node, idx)

"""Index helpers (.at[..] scatter/gather operations) for ArrayTree."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Self, cast

import equinox as eqx
import jax
import numpy as np

if TYPE_CHECKING:
    from pytree_utils.array_tree import ArrayTree


type AnyInt = int | np.integer
type StaticIndex = AnyInt | slice | type[Ellipsis]  # ty:ignore[invalid-type-form]
type Index = StaticIndex | None | Sequence[AnyInt] | jax.Array | np.ndarray


def _walk(root: Any, path: tuple[str, ...]) -> Any:
    """Follow an attribute *path* from *root* to the focused node or leaf."""
    obj = root
    for key in path:
        obj = getattr(obj, key)
    return obj


@dataclass(frozen=True)
class IndexedHelper[T: ArrayTree]:
    """Returned by ``ArrayTree.at[idx]``; mirrors JAX's scatter/gather API.

    *node* is always the original root tree. *path* is the attribute chain
    selected via ``.at.<field>...`` (empty when indexing the root directly), and
    *idx* is the captured index. Scatter ops operate on the focused subtree and
    splice the result back into the root, so they return *T*; ``get`` narrows the
    focus and returns that subtree's type instead.
    """

    node: T
    path: tuple[str, ...]
    idx: tuple

    def _focus(self) -> Any:
        """The node or leaf addressed by ``path`` within ``node``."""
        return _walk(self.node, self.path)

    def _splice(self, new_focus: Any) -> T:
        """Put *new_focus* back where ``path`` points, returning the root tree.

        The getter is rebuilt from the stored attribute ``path`` (not bound to
        the original focus object), so ``eqx.tree_at`` can locate the focus --
        whether it is a whole submodule or a single leaf -- inside ``node``.
        """
        if not self.path:
            return new_focus
        return eqx.tree_at(lambda root: _walk(root, self.path), self.node, new_focus)

    def _scatter(self, op: str, values: Any, **kwargs: Any) -> T:
        """Broadcast *values* to the focus, apply ``arr.at[idx].<op>``, splice back."""
        focus = self._focus()
        values_tree = jax.tree.broadcast(values, focus)
        new_focus = jax.tree.map(
            lambda arr, v: getattr(arr.at[self.idx], op)(v, **kwargs),
            focus,
            values_tree,
        )
        return self._splice(new_focus)

    def get(self, **kwargs: Any) -> Any:
        """Return the focused subtree with ``idx`` applied to every leaf.

        Unlike the scatter ops, ``get`` narrows (drops the indexed axes), so it
        returns the focus type rather than the root -- a leaf-targeted path
        yields a bare array, and the empty-path case still returns the root.
        Each node's ``_shape`` leaf is indexed alongside the data, so reported
        shapes stay correct without extra bookkeeping.
        """
        return jax.tree.map(lambda arr: arr.at[self.idx].get(**kwargs), self._focus())

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
        focus = self._focus()
        new_focus = jax.tree.map(lambda arr: arr.at[self.idx].apply(func, **kwargs), focus)
        return self._splice(new_focus)


@dataclass(frozen=True)
class IndexHelper[T: ArrayTree]:
    """Returned by ``ArrayTree.at``; walks a field path then captures an index.

    ``node.at[idx]`` indexes the root directly. ``node.at.field[idx]`` (any
    depth, nodes or leaves) focuses a sub-path: scatter ops splice their result
    back into the root, while ``get`` returns the narrowed focus.
    """

    node: T
    path: tuple[str, ...] = ()
    index: tuple[Index, ...] = ()

    def __getattr__(self, name: str) -> Self:
        if name.startswith("_"):
            raise AttributeError(name)

        focus = _walk(self.node, self.path)
        if not hasattr(focus, name):
            raise AttributeError(f"{type(focus)} has no field {name!r} to index into")

        index_pad = (slice(None),) * (focus.ndim - count_index_dims(self.index))
        return replace(self, path=(*self.path, name), index=self.index + index_pad)

    def __getitem__(self, idx: Index | Sequence[Index]) -> Self:
        normed_idx = cast(tuple[Index, ...], jax.numpy.index_exp[idx])
        index: tuple[Index, ...] = self.index + normed_idx
        ndim = count_index_dims(index)

        focus = _walk(self.node, self.path)
        if ndim > focus.ndim:
            where = ".".join(("self", *self.path))
            raise IndexError(
                f"Too many indicies: node {where} is {focus.ndim}-dimensional but {ndim} were indexed"
            )

        return replace(self, index=index)


def count_index_dims(idx: tuple) -> int:
    """Count how many existing dimensions an index tuple addresses.

    Every element other than ``None`` / ``numpy.newaxis`` addresses one
    dimension.
    """
    return sum(i is not None for i in idx)

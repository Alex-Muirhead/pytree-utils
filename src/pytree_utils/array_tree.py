"""The ArrayTree base class."""

from collections.abc import Sequence
from typing import Self, TypeGuard

import equinox as eqx
import jax

from pytree_utils.array_methods import ArrayTreeOps
from pytree_utils.indexing import Index, IndexHelper


@jax.tree_util.register_pytree_node_class
class ShapeMarker(tuple):
    """PyTree Node to track shape transformations.

    When flattened, will produce a zero-sized array carrying the shape.
    Any JAX transformation will on this leaf will be tracked in the shape.
    For all python-purposes, this is a nice tuple!
    """

    __slots__ = ()

    def __new__(cls, shape: int | Sequence[int]) -> Self:
        """Construct the ShapeMarker."""
        if not isinstance(shape, Sequence):
            shape = (shape,)
        return super().__new__(cls, shape)

    def __init__(self, _: int | Sequence[int]) -> None:
        """Construct the ShapeMarker."""
        # We do the validation here, to avoid any costs and runtime pains in __new__
        for s in self:
            if not isinstance(s, int):
                raise TypeError("Shape must be integer values")
            if s < 0:
                raise ValueError("All shape values must be >= 0")

    def tree_flatten(self) -> tuple[tuple[jax.Array], tuple]:
        """JAX ``flatten`` function for ShapeMarker."""
        # NOTE: Do a prefix for now, then work out later if it's worth it
        ghost = jax.numpy.empty((0, *self))
        children = (ghost,)
        aux_data = ()
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: tuple, children: tuple[jax.Array]) -> Self:
        """JAX ``unflatten`` function for ShapeMarker."""
        del aux_data
        (ghost,) = children
        _, *shape = ghost.shape
        return cls(shape)

    @classmethod
    def isinstance(cls, value: object) -> TypeGuard[Self]:
        """Shorthand function to check if a value is a ShapeMarker."""
        return isinstance(value, cls)


class ArrayTree(ArrayTreeOps, eqx.Module):
    """PyTree node with array semantics passed to leaves."""

    shape: ShapeMarker = eqx.field(converter=ShapeMarker, kw_only=True)

    @property
    def ndim(self) -> int:
        """Number of dimensions of this node."""
        return len(self.shape)

    def __check_init__(self) -> None:
        """Validate that every leaf is prefixed by this node's accumulated shape."""
        for path, leaf in jax.tree.leaves_with_path(self, is_leaf=ShapeMarker.isinstance):
            if ShapeMarker.isinstance(leaf):
                continue

            name = jax.tree_util.keystr(path)

            if not isinstance(leaf, jax.Array):
                raise TypeError(f"Bad leaf type at {name}\nExpected jax.Array type, got {type(leaf)}")

            if leaf.shape[: self.ndim] != self.shape:
                raise ValueError(
                    f"Bad leaf shape at {name}\nExpected shape prefix with {self.shape}, got {leaf.shape}"
                )

    @property
    def at(self) -> IndexHelper[Self]:
        """Entry point for prefix-validated indexing.

        Use ``node.at[i].get()`` / ``node.at[i].set(v)``, or focus a field path
        first with ``node.at.field[i].set(v)`` to update in place while keeping
        the root type.
        """
        return IndexHelper(self)

    def __getitem__(self, idx: Index | tuple[Index]) -> Self:
        """Get index on each leaf."""
        return self.at[idx].get()

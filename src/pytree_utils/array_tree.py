"""The ArrayTree base class."""

from typing import Self

import equinox as eqx
import jax

from pytree_utils._index import IndexHelper
from pytree_utils._ops import ArrayTreeOps
from pytree_utils._spec import ShapeType


class ArrayTree(ArrayTreeOps, eqx.Module):
    # Marker leaf: used for tracking the shape of this node under transformations
    # Array prefixed with shape (0,) to ensure no memory allocation
    # All dynamic operations should be no-op
    __ghost: jax.Array = eqx.field(kw_only=True, repr=False)

    @property
    def shape(self) -> ShapeType:
        """Shape of this node."""
        return self.__ghost.shape[1:]

    @property
    def ndim(self) -> int:
        """Number of dimensions of this node."""
        return len(self.__ghost) - 1

    def __check_init__(self):
        """Validate that every leaf is postfixed by this node's accumulated shape."""
        suffix = self.shape
        ndim = len(suffix)
        for path, leaf in jax.tree.leaves_with_path(self):
            name = "self" + jax.tree_util.keystr(path)
            if not isinstance(leaf, jax.Array):
                raise TypeError(f"Bad leaf type at {name}\nExpected jax.Array type, got {type(leaf)}")
            if leaf.shape[-ndim:] != suffix:
                raise ValueError(
                    f"Bad leaf shape at {name}\nExpected shape suffixed with {suffix}, got {leaf.shape}"
                )

    @property
    def at(self) -> IndexHelper[Self]:
        """Entry point for prefix-validated indexing.

        Use ``node.at[i].get()`` / ``node.at[i].set(v)``, or focus a field path
        first with ``node.at.field[i].set(v)`` to update in place while keeping
        the root type.
        """
        return IndexHelper(self)

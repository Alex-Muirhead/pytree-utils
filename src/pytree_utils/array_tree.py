"""The ArrayTree base class."""

import functools
from typing import Any, Self

import equinox as eqx
import jax
import jax.numpy as jnp

from pytree_utils._index import IndexHelper
from pytree_utils._ops import ArrayTreeOps
from pytree_utils._spec import ShapeType


class ArrayTree(ArrayTreeOps, eqx.Module):
    """An immutable, nested struct-of-arrays that is a valid JAX pytree.

    Define a structure by subclassing ``ArrayTree`` and declaring array leaves
    with ``leaf()`` and child nodes with ``node(shape=...)``::

        class Vel(ArrayTree):
            vx: jax.Array = leaf(shape=(1,))
            vy: jax.Array = leaf(shape=(2,))

        class World(ArrayTree):
            vel: Vel = node(shape=(3,))

    Instances are not constructed directly. ``blueprint(cls, shape=...)``
    returns a mutable blueprint describing the structure before any arrays are
    allocated; edit its fields freely -- including overriding node shapes --
    then call ``.zeros()`` / ``.ones()`` / ``.empty()`` /
    ``.full(fill_value=...)`` to allocate the tree::

        proto = blueprint(World, shape=(2,))
        proto.vel.shape = (4,)        # override the node's block shape
        world = proto.zeros()

        world.vel.vx.shape            # (2, 4, 1) -- World(2) + Vel(4) + leaf(1)
        world.vel.shape               # (2, 4)    -- node's full accumulated shape

    Generic nodes are parameterised with the standard ``Cls[T]`` syntax, e.g.
    ``blueprint(Container[Vel], shape=(2,))``.

    **Indexing**
    ``.at[idx].get()`` / ``.at[idx].set(values)`` index into the accumulated
    prefix; indices that reach into a leaf's own dimensions are rejected::

        world.at[0].get()           # ok -- World prefix is 1-dim
        world.at[0, 1].get()        # IndexError
        world.vel.at[0, 3].get()    # ok -- vel's accumulated prefix is (2, 4)
    """

    # Zero-sized array (trailing axis 0) whose remaining axes are this node's
    # accumulated prefix. Costs no memory, but as a real leaf it is sliced in
    # lockstep with the data under indexing, so ``shape`` needs no bookkeeping.
    _shape: jax.Array = eqx.field(kw_only=True, repr=False)

    @property
    def shape(self) -> ShapeType:
        """Full accumulated shape of this node (every ancestor block plus its own).

        Read straight off the dedicated ``_shape`` leaf by dropping its empty
        trailing axis. Because that leaf rides along with the data under any
        indexing, the reported shape stays correct with no metadata to rewrite.
        """
        return self._shape.shape[:-1]

    def __check_init__(self):
        """Validate that every leaf is prefixed by this node's accumulated shape."""
        prefix = self.shape
        n = len(prefix)
        for path, leaf in jax.tree.leaves_with_path(self):
            if leaf.shape[:n] != prefix:
                raise ValueError(
                    f"Bad leaf shape at self{jax.tree_util.keystr(path)}\n"
                    f"Expected shape prefixed with {prefix}, got {leaf.shape}"
                )

    @property
    def at(self) -> IndexHelper[Self]:
        """Entry point for prefix-validated indexing.

        Use ``node.at[i].get()`` or ``node.at[i].set(v)``.
        """
        return IndexHelper(self)

    def zeros_like(self) -> Self:
        """Return a copy with all arrays replaced by zeros of the same shape."""
        return jax.tree.map(jnp.zeros_like, self)

    def ones_like(self) -> Self:
        """Return a copy with all arrays replaced by ones of the same shape."""
        return jax.tree.map(jnp.ones_like, self)

    def full_like(self, fill_value: Any) -> Self:
        """Return a copy with all arrays replaced by fill_value of the same shape."""
        init_fn = functools.partial(jnp.full_like, fill_value=fill_value)
        return jax.tree.map(init_fn, self)

    def empty_like(self) -> Self:
        """Return a copy with all arrays replaced by uninitialised arrays."""
        return jax.tree.map(jnp.empty_like, self)

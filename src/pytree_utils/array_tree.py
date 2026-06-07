"""The ArrayTree base class."""

import functools
from typing import Any, Self

import equinox as eqx
import jax
import jax.numpy as jnp

from pytree_utils._index import _IndexHelper
from pytree_utils._ops import _ArrayTreeOps
from pytree_utils._spec import ShapeType


class ArrayTree(_ArrayTreeOps, eqx.Module):
    """Base class for instantiated array structures (Stage 3).

    ``ArrayTree`` instances are immutable equinox modules and valid JAX
    pytrees. They are produced exclusively by building a ``Blueprint``
    (Stage 2):

    **Stage 1 - Definition**
    Subclass ``ArrayTree`` and declare array leaves with ``leaf()`` and child
    nodes with ``node(shape=...)``. Every node's nominal shape is ``()``
    unless given by its parent::

        class Vel(ArrayTree):
            vx: jax.Array = leaf(shape=(1,))
            vy: jax.Array = leaf(shape=(2,))

        class World(ArrayTree):
            vel: Vel = node(shape=(3,))

    **Stage 2 - Blueprint (mutable)**
    Call ``blueprint(cls, shape=...)`` to get a mutable ``Blueprint`` whose
    fields can be edited freely before any arrays are allocated. Generic
    nodes are parameterised with the standard ``Cls[T]`` syntax::

        proto = blueprint(World, shape=(2,))
        proto.vel.shape = (4,)                # direct mutation
        proto.vel = blueprint(Vel, shape=(5,))  # swap child
        cproto = blueprint(Container[Vel], shape=(2,))  # generic

    **Stage 3 - Instantiation (immutable pytree)**
    Call ``.zeros()``, ``.ones()``, ``empty()``, or ``.full(fill_value=...)``
    on the blueprint to produce a real ``ArrayTree``::

        world = proto.zeros()
        world.vel.vx.shape  # (2, 5, 1) -- World(2) + Vel(5) + leaf(1)
        world.vel.shape     # (2, 5)    -- node's full accumulated shape

    **Indexing**
    Use ``.at[idx].get()`` / ``.at[idx].set(values)`` to index into the
    accumulated prefix. Indices that reach into leaf-specific dimensions
    are rejected::

        world.at[0].get()           # ok -- World prefix is 1-dim
        world.at[0, 1].get()        # IndexError
        world.vel.at[0, 3].get()    # ok -- vel's accumulated prefix is (2, 5)
    """

    # A dedicated zero-sized array whose non-trailing axes spell out this node's
    # full accumulated prefix. It costs no memory (its trailing axis is 0) yet,
    # being a real leaf, it is sliced in lockstep with the data leaves whenever
    # the tree is indexed -- so ``shape`` never needs separate bookkeeping.
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
    def at(self) -> _IndexHelper[Self]:
        """Entry point for prefix-validated indexing.

        Use ``node.at[i].get()`` or ``node.at[i].set(v)``.
        """
        return _IndexHelper(self)

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

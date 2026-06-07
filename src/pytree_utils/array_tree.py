"""The ArrayTree base class."""

import dataclasses as dc
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

    _own_shape: ShapeType = eqx.field(static=True, kw_only=True, default=(), repr=False)

    @property
    def shape(self) -> ShapeType:
        """Full accumulated shape of this node (every ancestor block plus its own).

        Derived on demand rather than stored: a representative array-bearing
        field reveals the prefix once its own (indexing-invariant) shape is
        stripped off. Because of this, indexing only has to slice the leaf
        arrays -- the reported shape then follows automatically, with no
        per-node metadata to rewrite.
        """
        for f in dc.fields(self):
            prefix = self._field_prefix(f)
            if prefix is not None:
                return prefix
        return self._own_shape

    def _field_prefix(self, f: dc.Field) -> ShapeType | None:
        """Accumulated prefix implied by field *f*, or None if it holds no array data.

        For a leaf field the prefix is the array shape minus the leaf's own
        (fixed) shape; for a child node it is the child's full shape minus the
        child's own block. Both subtrahends are invariant under indexing.
        """
        if not f.init or f.metadata.get("static", False):
            return None
        val = getattr(self, f.name)
        if isinstance(val, jax.Array):
            spec = f.metadata.get("leaf_spec")
            if spec is None:
                return None
            return val.shape[: val.ndim - len(spec.shape)]
        if isinstance(val, ArrayTree):
            child = val.shape
            return child[: len(child) - len(val._own_shape)]
        return None

    def __check_init__(self):
        """Validate that every array-bearing field agrees on the accumulated prefix."""
        prefixes = [p for f in dc.fields(self) if (p := self._field_prefix(f)) is not None]
        if prefixes and any(p != prefixes[0] for p in prefixes):
            raise ValueError(
                f"Inconsistent accumulated prefixes across fields of "
                f"{type(self).__name__}: {prefixes}"
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

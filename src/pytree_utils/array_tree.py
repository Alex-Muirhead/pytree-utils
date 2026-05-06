"""The ArrayTree base class."""

import dataclasses as dc
import functools
from typing import Any, Self

import equinox as eqx
import jax
import jax.numpy as jnp

from pytree_utils._blueprint import (
    _BlueprintBase,
    _get_blueprint_cls,
    _ParameterizedTree,
)
from pytree_utils._index import _IndexHelper
from pytree_utils._ops import _ArrayTreeOps
from pytree_utils._spec import ShapeInput, ShapeType, _to_shape


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
    Call ``cls.blueprint(shape=...)`` to get a mutable ``Blueprint`` whose
    fields can be edited freely before any arrays are allocated::

        proto = World.blueprint(shape=(2,))
        proto.vel.shape = (4,)            # direct mutation
        proto.vel = Vel.blueprint(shape=(5,))  # swap child

    **Stage 3 - Instantiation (immutable pytree)**
    Call ``.zeros()``, ``.ones()``, ``empty()``, or ``.full(fill_value=...)``
    on the blueprint to produce a real ``ArrayTree``::

        world = proto.zeros()
        world.vel.vx.shape  # (2, 5, 1) — World(2) + Vel(5) + leaf(1)

    **Indexing**
    Use ``.at[idx].get()`` / ``.at[idx].set(values)`` to index into the
    accumulated prefix. Indices that reach into leaf-specific dimensions
    are rejected::

        world.at[0].get()           # ok — World prefix is 1-dim
        world.at[0, 1].get()        # IndexError
        world.vel.at[0, 3].get()    # ok — vel's accumulated prefix is (2, 5)
    """

    shape: ShapeType = eqx.field(static=True, kw_only=True, default=())
    _prefix: ShapeType = eqx.field(static=True, kw_only=True, default=(), repr=False)

    def __check_init__(self):
        """Validate the ArrayTree invariants."""
        full_prefix = self._prefix + self.shape

        for path, leaf in jax.tree.leaves_with_path(self):
            if leaf.shape[: len(full_prefix)] != full_prefix:
                raise ValueError(
                    f"Bad leaf shape at self{jax.tree_util.keystr(path)}\n"
                    f"Expected shape prefixed with {full_prefix}, got {leaf.shape}"
                )

    @classmethod
    def __class_getitem__(cls, params: Any) -> _ParameterizedTree:
        """Support ``GenericNode[ConcreteType].blueprint(...)`` syntax."""
        if not isinstance(params, tuple):
            params = (params,)
        type_params = getattr(cls, "__type_params__", ())
        if not type_params:
            return super().__class_getitem__(params if len(params) > 1 else params[0])  # type: ignore[misc]
        return _ParameterizedTree(cls, dict(zip(type_params, params, strict=False)))

    @classmethod
    def blueprint(cls, shape: ShapeInput = ()) -> _BlueprintBase:
        """Create a mutable Blueprint for this node type (Stage 2)."""
        return _get_blueprint_cls(cls)(shape=_to_shape(shape))

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

    def _reindex(
        self,
        idx: tuple,
        new_prefix: ShapeType,
        new_shape: ShapeType,
        get_kw: dict | None = None,
    ) -> Self:
        """Apply *idx* to all leaf arrays and update prefix/shape metadata."""
        init_kw: dict[str, Any] = {"_prefix": new_prefix, "shape": new_shape}
        _gkw: dict[str, Any] = get_kw or {}

        for f in dc.fields(self):
            if not f.init or f.name in ("_prefix", "shape"):
                continue
            if f.metadata.get("static", False):
                init_kw[f.name] = getattr(self, f.name)
                continue

            val = getattr(self, f.name)
            if isinstance(val, jax.Array):
                init_kw[f.name] = val.at[idx].get(**_gkw)
            elif isinstance(val, ArrayTree):
                init_kw[f.name] = val._reindex(idx, new_prefix + new_shape, val.shape, get_kw)
            else:
                init_kw[f.name] = val

        return type(self)(**init_kw)

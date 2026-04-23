from __future__ import annotations

import dataclasses as dc
import functools
import typing
from typing import Any, Callable, ClassVar, Self

import equinox as eqx
import jax
import jax.numpy as jnp

ShapeType = tuple[int, ...]
InitFn = Callable[[ShapeType, Any], jax.Array]


# ---------------------------------------------------------------------------
# LeafSpec — describes one array leaf in a Blueprint
# ---------------------------------------------------------------------------


@dc.dataclass(frozen=True)
class LeafSpec:
    """Shape/dtype specification for an array leaf.

    Lives inside a ``Blueprint`` (Stage 2) and is replaced by a real
    ``jax.Array`` when the Blueprint is built (Stage 3).
    """

    shape: ShapeType
    dtype: Any = float


# ---------------------------------------------------------------------------
# Field helpers
# ---------------------------------------------------------------------------


def leaf(shape: ShapeType = (), dtype: Any = float, **kwargs) -> dc.Field:
    """Declare an array leaf field with shape and dtype (Stage 1).

    Usage::

        class MyNode(ArrayTree):
            x: jax.Array = leaf(shape=(3,))
            y: jax.Array = leaf(shape=(4,), dtype=jnp.float16)
    """
    metadata = dict(kwargs.pop("metadata", None) or {})
    if "leaf_spec" in metadata:
        raise ValueError("leaf_spec multiply defined in metadata")
    metadata["leaf_spec"] = LeafSpec(shape=shape, dtype=dtype)
    return dc.field(**kwargs, metadata=metadata)


def node(shape: ShapeType = (), **kwargs) -> dc.Field:
    """Declare a child node field and specify its shape (Stage 1).

    The *shape* is used when the parent's ``Blueprint`` is constructed,
    so the node class itself needs no hard-coded shape::

        class Vel(ArrayTree):
            vx: jax.Array = leaf(shape=(1,))
            vy: jax.Array = leaf(shape=(2,))

        class World(ArrayTree):
            vel: Vel = node(shape=(3,))

        proto = World.blueprint(shape=(2,))
        # proto.shape == (2,), proto.vel.shape == (3,)
    """
    metadata = dict(kwargs.pop("metadata", None) or {})
    if "node_shape" in metadata:
        raise ValueError("node_shape multiply defined in metadata")
    metadata["node_shape"] = shape
    return dc.field(**kwargs, metadata=metadata)


def _field_default(field: dc.Field, *, throw: bool = True) -> Any:
    """Return the default value for a dataclass field."""
    if field.default is not dc.MISSING:
        return field.default
    if field.default_factory is not dc.MISSING:
        return field.default_factory()
    if throw:
        raise ValueError(f"Field '{field.name}' has no default value or factory")
    return None


def _count_index_dims(idx: tuple) -> int:
    """Count how many existing dimensions are addressed by an index tuple.

    Each element other than ``None`` / ``numpy.newaxis`` addresses one
    dimension.
    """
    return sum(i is not None for i in idx)


# ---------------------------------------------------------------------------
# Index helpers  (.at[idx].get() / .at[idx].set())
# ---------------------------------------------------------------------------


@dc.dataclass(frozen=True)
class _IndexedHelper:
    """Returned by ``ArrayTree.at[idx]``; provides ``.get()`` and ``.set()``."""

    _node: ArrayTree
    _idx: tuple

    def get(self) -> ArrayTree:
        """Return a new node with ``idx`` applied to all leaf arrays."""
        full_prefix = self._node._prefix + self._node.shape
        remaining = full_prefix[_count_index_dims(self._idx) :]
        return self._node._reindex(self._idx, new_prefix=(), new_shape=remaining)

    def set(self, values: Any) -> ArrayTree:
        """Return a copy of the node with indexed leaves updated from *values*.

        *values* is broadcast to the node's structure via ``jax.tree.broadcast``:
        a scalar fills every leaf; a same-structure ``ArrayTree`` sets each leaf
        from the corresponding leaf in *values*.
        """
        idx = self._idx
        values_tree = jax.tree.broadcast(values, self._node)
        return jax.tree.map(lambda arr, v: arr.at[idx].set(v), self._node, values_tree)


@dc.dataclass(frozen=True)
class _IndexHelper:
    """Returned by ``ArrayTree.at``; validates the index and captures it."""

    _node: ArrayTree

    def __getitem__(self, idx: Any) -> _IndexedHelper:
        if not isinstance(idx, tuple):
            idx = (idx,)

        full_prefix = self._node._prefix + self._node.shape
        n = _count_index_dims(idx)

        if n > len(full_prefix):
            raise IndexError(
                f"{type(self._node).__name__} has a {len(full_prefix)}-dimensional "
                f"prefix {full_prefix}; cannot index with {n} dimension(s)"
            )

        return _IndexedHelper(self._node, idx)


# ---------------------------------------------------------------------------
# Blueprint — mutable Stage-2 structure
# ---------------------------------------------------------------------------


class _BlueprintBase[T: ArrayTree]:
    """Base for all generated Blueprint types.

    Blueprints are plain mutable Python objects (not JAX pytrees).  They
    describe the shape/dtype structure of an ``ArrayTree`` before any arrays
    are allocated.  Mutate fields directly, then call ``.zeros()`` or
    ``.ones()`` to produce a fully instantiated ``ArrayTree``.
    """

    __slots__ = ()

    _array_tree_cls: ClassVar[type]

    def build(self, prefix: ShapeType = (), init_fn: InitFn = jnp.zeros) -> T:
        """Instantiate arrays from this blueprint (Stage 3).

        Args:
            prefix: Extra leading dimensions prepended outside this node's
                    own ``shape``.
            init_fn: ``(shape, dtype) -> jax.Array``.  Defaults to
                     ``jnp.zeros``.
        """
        cls = self._array_tree_cls
        accumulated = prefix + self.shape
        kwargs: dict[str, Any] = {"shape": self.shape, "_prefix": prefix}

        for f in dc.fields(cls):
            if not f.init or f.name in ("shape", "_prefix"):
                continue
            if f.metadata.get("static", False):
                kwargs[f.name] = _field_default(f)
                continue

            val = getattr(self, f.name)
            if isinstance(val, LeafSpec):
                kwargs[f.name] = init_fn(accumulated + val.shape, val.dtype)
            elif isinstance(val, _BlueprintBase):
                kwargs[f.name] = val.build(prefix=accumulated, init_fn=init_fn)
            else:
                kwargs[f.name] = val

        return cls(**kwargs)

    def zeros(self, prefix: ShapeType = ()) -> T:
        """Build with zero-filled arrays."""
        return self.build(prefix=prefix, init_fn=jnp.zeros)

    def ones(self, prefix: ShapeType = ()) -> T:
        """Build with one-filled arrays."""
        return self.build(prefix=prefix, init_fn=jnp.ones)


def _get_blueprint_cls(array_tree_cls: type) -> type:
    """Return (creating if necessary) the Blueprint class for an ArrayTree subclass."""
    if "_blueprint_cls" not in array_tree_cls.__dict__:
        array_tree_cls._blueprint_cls = _make_blueprint_cls(array_tree_cls)
    return array_tree_cls._blueprint_cls


def _make_blueprint_cls(array_tree_cls: type) -> type:
    """Dynamically generate a mutable Blueprint dataclass for *array_tree_cls*."""
    fields: list = [("shape", ShapeType, dc.field(default=()))]
    hints = typing.get_type_hints(array_tree_cls)

    for f in dc.fields(array_tree_cls):
        if f.name in ("shape", "_prefix") or not f.init:
            continue
        if f.metadata.get("static", False):
            continue

        if "leaf_spec" in f.metadata:
            fields.append((f.name, LeafSpec, dc.field(default=f.metadata["leaf_spec"])))
        else:
            hint = hints.get(f.name)
            origin: Any = typing.get_origin(hint) or hint
            if isinstance(origin, type) and issubclass(origin, ArrayTree):
                child_bp_cls = _get_blueprint_cls(origin)
                node_shape = f.metadata.get("node_shape", ())
                factory = functools.partial(child_bp_cls, shape=node_shape)
                fields.append((f.name, child_bp_cls, dc.field(default_factory=factory)))

    blueprint_cls = dc.make_dataclass(
        f"{array_tree_cls.__name__}Blueprint",
        fields,
        bases=(_BlueprintBase[array_tree_cls],),
        slots=True,
    )
    blueprint_cls._array_tree_cls = array_tree_cls
    return blueprint_cls


# ---------------------------------------------------------------------------
# ArrayTree — immutable Stage-3 pytree
# ---------------------------------------------------------------------------


class ArrayTree(eqx.Module):
    """Base class for instantiated array structures (Stage 3).

    ``ArrayTree`` instances are immutable equinox modules and valid JAX
    pytrees.  They are produced exclusively by building a ``Blueprint``
    (Stage 2):

    **Stage 1 – Definition**
    Subclass ``ArrayTree`` and declare array leaves with ``leaf()`` and child
    nodes with ``node(shape=...)``.  Every node's nominal shape is ``()``
    unless given by its parent::

        class Vel(ArrayTree):
            vx: jax.Array = leaf(shape=(1,))
            vy: jax.Array = leaf(shape=(2,))

        class World(ArrayTree):
            vel: Vel = node(shape=(3,))

    **Stage 2 – Blueprint (mutable)**
    Call ``cls.blueprint(shape=...)`` to get a mutable ``Blueprint`` whose
    fields can be edited freely before any arrays are allocated::

        proto = World.blueprint(shape=(2,))
        proto.vel.shape = (4,)            # direct mutation
        proto.vel = Vel.blueprint(shape=(5,))  # swap child

    **Stage 3 – Instantiation (immutable pytree)**
    Call ``.zeros()``, ``.ones()``, or ``.build(init_fn=...)`` on the
    blueprint to produce a real ``ArrayTree``::

        world = proto.zeros()
        world.vel.vx.shape  # (2, 5, 1) — World(2) + Vel(5) + leaf(1)

    **Indexing**
    Use ``.at[idx].get()`` / ``.at[idx].set(values)`` to index into the
    accumulated prefix.  Indices that reach into leaf-specific dimensions
    are rejected::

        world.at[0].get()           # ok — World prefix is 1-dim
        world.at[0, 1].get()        # IndexError
        world.vel.at[0, 3].get()    # ok — vel's accumulated prefix is (2, 5)
    """

    shape: ShapeType = eqx.field(static=True, kw_only=True, default=())
    _prefix: ShapeType = eqx.field(static=True, kw_only=True, default=(), repr=False)

    @classmethod
    def blueprint(cls, shape: ShapeType = ()) -> _BlueprintBase[Self]:
        """Create a mutable Blueprint for this node type (Stage 2)."""
        return _get_blueprint_cls(cls)(shape=shape)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    @property
    def at(self) -> _IndexHelper:
        """Entry point for prefix-validated indexing: ``node.at[i].get()`` / ``.set(v)``."""
        return _IndexHelper(self)

    # ------------------------------------------------------------------
    # Utilities (Stage 3 only — leaves must be real jax.Arrays)
    # ------------------------------------------------------------------

    def zeros_like(self) -> Self:
        """Return a copy with all arrays replaced by zeros of the same shape."""
        return jax.tree.map(jnp.zeros_like, self)

    def ones_like(self) -> Self:
        """Return a copy with all arrays replaced by ones of the same shape."""
        return jax.tree.map(jnp.ones_like, self)

    def empty_like(self) -> Self:
        """Return a copy with all arrays replaced by uninitialised arrays."""
        return jax.tree.map(jnp.empty_like, self)

    # ------------------------------------------------------------------
    # Private helpers for .at indexing
    # ------------------------------------------------------------------

    def _reindex(self, idx: tuple, new_prefix: ShapeType, new_shape: ShapeType) -> Self:
        """Apply *idx* to all leaf arrays and update prefix/shape metadata."""
        kwargs: dict[str, Any] = {"_prefix": new_prefix, "shape": new_shape}

        for f in dc.fields(self):
            if not f.init or f.name in ("_prefix", "shape"):
                continue
            if f.metadata.get("static", False):
                kwargs[f.name] = getattr(self, f.name)
                continue

            val = getattr(self, f.name)
            if isinstance(val, jax.Array):
                kwargs[f.name] = val[idx]
            elif isinstance(val, ArrayTree):
                kwargs[f.name] = val._reindex(idx, new_prefix + new_shape, val.shape)
            else:
                kwargs[f.name] = val

        return type(self)(**kwargs)


# ---------------------------------------------------------------------------
# Example / smoke test
# ---------------------------------------------------------------------------


class Vel(ArrayTree):
    vx: jax.Array = leaf(shape=(1,))
    vy: jax.Array = leaf(shape=(2,))


class World(ArrayTree):
    vel: Vel = node(shape=(3,))


if __name__ == "__main__":
    # Stage 2: build a mutable Blueprint
    proto = World.blueprint(shape=(2,))
    print("Stage 2 – blueprint:")
    print(proto)

    # Direct mutation — no eqx.tree_at needed
    proto.vel.shape = (4,)

    # Stage 3: instantiate from blueprint
    world = proto.zeros()
    print("\nStage 3 – instantiated (zeros):")
    print(world)
    print("vel.vx shape:", world.vel.vx.shape)  # (2, 4, 1)
    print("vel.vy shape:", world.vel.vy.shape)  # (2, 4, 2)

    # Indexing (Stage 3)
    print("\nIndexing:")
    s0 = world.at[0].get()
    print("world.at[0].get() — vel.vx shape:", s0.vel.vx.shape)  # (4, 1)
    s03 = world.vel.at[0, 3].get()
    print("world.vel.at[0,3].get() — vx shape:", s03.vx.shape)  # (1,)
    print("world.vel.at[0,3].get() — vy shape:", s03.vy.shape)  # (2,)

    # Broadcast set (Stage 3)
    world2 = world.at[0].set(1.0)
    print("\nworld.at[0].set(1.0) — vel.vx[0,0,0]:", world2.vel.vx[0, 0, 0])  # 1.0
    print("world.at[0].set(1.0) — vel.vx[1,0,0]:", world2.vel.vx[1, 0, 0])  # 0.0

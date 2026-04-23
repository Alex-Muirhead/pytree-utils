"""Blueprint machinery, index helpers, and the ArrayTree base class."""

from __future__ import annotations

import dataclasses as dc
import functools
import typing
from collections.abc import Callable
from typing import Any, ClassVar, Self

import equinox as eqx
import jax
import jax.numpy as jnp

from pytree_utils._ops import _ArrayTreeOps
from pytree_utils._spec import (
    InitFn,
    LeafSpec,
    ShapeInput,
    ShapeType,
    _count_index_dims,
    _field_default,
    _to_shape,
)

# ---------------------------------------------------------------------------
# Blueprint — mutable Stage-2 structure
# ---------------------------------------------------------------------------


class _BlueprintBase[T: ArrayTree]:
    """Base for all generated Blueprint types.

    Blueprints are plain mutable Python objects (not JAX pytrees). They
    describe the shape/dtype structure of an ``ArrayTree`` before any arrays
    are allocated. Mutate fields directly, then call ``.zeros()`` or
    ``.ones()`` to produce a fully instantiated ``ArrayTree``.
    """

    __slots__ = ()

    _array_tree_cls: ClassVar[type]

    def _build(self, prefix: ShapeInput = (), init_fn: InitFn = jnp.zeros) -> T:
        """Instantiate arrays from this blueprint (Stage 3).

        Args:
            prefix: Extra leading dimensions prepended outside this node's
                    own ``shape``.
            init_fn: ``(shape, dtype) -> jax.Array``.  Defaults to
                     ``jnp.zeros``.
        """
        cls = self._array_tree_cls
        prefix = _to_shape(prefix)
        accumulated = prefix + self.shape  # type: ignore[attr-defined]
        kwargs: dict[str, Any] = {
            "shape": self.shape,  # type: ignore[attr-defined]
            "_prefix": prefix,
        }

        for f in dc.fields(cls):
            if not f.init or f.name in ("shape", "_prefix"):
                continue
            if f.metadata.get("static", False):
                kwargs[f.name] = _field_default(f)
                continue

            val = getattr(self, f.name)
            if isinstance(val, LeafSpec):
                kwargs[f.name] = init_fn(accumulated + val.shape, dtype=val.dtype)
            elif isinstance(val, _BlueprintBase):
                kwargs[f.name] = val._build(prefix=accumulated, init_fn=init_fn)
            else:
                kwargs[f.name] = val

        return cls(**kwargs)

    def zeros(self, prefix: ShapeInput = ()) -> T:
        """Build with zero-filled arrays."""
        return self._build(prefix=prefix, init_fn=jnp.zeros)

    def ones(self, prefix: ShapeInput = ()) -> T:
        """Build with one-filled arrays."""
        return self._build(prefix=prefix, init_fn=jnp.ones)

    def full(self, fill_value: Any, prefix: ShapeInput = ()) -> T:
        """Build with value-filled arrays."""
        init_fn = functools.partial(jnp.full, fill_value=fill_value)
        return self._build(prefix=prefix, init_fn=init_fn)

    def empty(self, prefix: ShapeInput = ()) -> T:
        """Build with empty arrays."""
        return self._build(prefix=prefix, init_fn=jnp.empty)


# ---------------------------------------------------------------------------
# Parameterized generic support  (Container[Vel].blueprint(...))
# ---------------------------------------------------------------------------


@dc.dataclass(frozen=True)
class _ParameterizedTree:
    """Returned by ``GenericArrayTree[ConcreteType]``; provides ``.blueprint()``."""

    cls: type
    type_map: dict

    def blueprint(self, shape: ShapeInput = ()) -> _BlueprintBase:
        """Create a Blueprint with TypeVars resolved to their concrete types."""
        return _make_blueprint_cls(self.cls, self.type_map)(shape=_to_shape(shape))


# ---------------------------------------------------------------------------
# Blueprint class factory
# ---------------------------------------------------------------------------


def _resolve_hint(hint: Any, type_map: dict) -> Any:
    """Substitute a TypeVar with its concrete type if present in *type_map*."""
    if isinstance(hint, typing.TypeVar):
        return type_map.get(hint, hint)
    return hint


def _get_blueprint_cls(array_tree_cls: type) -> type:
    """Return (creating if necessary) the Blueprint class for an ArrayTree subclass."""
    if "_blueprint_cls" not in array_tree_cls.__dict__:
        array_tree_cls._blueprint_cls = _make_blueprint_cls(array_tree_cls, {})  # type: ignore[attr-defined]
    return array_tree_cls._blueprint_cls  # type: ignore[attr-defined]


def _make_blueprint_cls(array_tree_cls: type, type_map: dict) -> type:
    """Dynamically generate a mutable Blueprint dataclass for *array_tree_cls*.

    *type_map* maps ``TypeVar`` objects to concrete ``ArrayTree`` subclasses,
    used when the class is generic (e.g. ``Container[Vel]``).
    """
    fields: list = [("shape", ShapeType, dc.field(default=()))]
    hints = typing.get_type_hints(array_tree_cls)

    for f in dc.fields(array_tree_cls):
        if f.name in ("shape", "_prefix") or not f.init:
            continue
        if f.metadata.get("static", False):
            continue

        if "leaf_spec" in f.metadata:
            fields.append((f.name, LeafSpec, dc.field(default=f.metadata["leaf_spec"])))
            continue

        hint = _resolve_hint(hints.get(f.name), type_map)
        node_shape = f.metadata.get("node_shape", ())

        if isinstance(hint, _ParameterizedTree):
            # e.g. field: Container[Pos] — annotation resolved to a _ParameterizedTree
            child_bp_cls = _make_blueprint_cls(hint.cls, hint.type_map)
            factory = functools.partial(child_bp_cls, shape=node_shape)
            fields.append((f.name, child_bp_cls, dc.field(default_factory=factory)))
        else:
            origin: Any = typing.get_origin(hint) or hint
            if isinstance(origin, type) and issubclass(origin, ArrayTree):
                child_bp_cls = _get_blueprint_cls(origin)
                factory = functools.partial(child_bp_cls, shape=node_shape)
                fields.append((f.name, child_bp_cls, dc.field(default_factory=factory)))

    blueprint_cls = dc.make_dataclass(
        f"{array_tree_cls.__name__}Blueprint",
        fields,
        bases=(_BlueprintBase[array_tree_cls],),
        slots=True,
    )
    blueprint_cls._array_tree_cls = array_tree_cls  # type: ignore[attr-defined]
    return blueprint_cls


# ---------------------------------------------------------------------------
# Index helpers  (.at[idx] scatter/gather operations)
# ---------------------------------------------------------------------------


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
        return self.node._reindex(
            self.idx, new_prefix=(), new_shape=remaining, get_kw=kwargs or None
        )

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


# ---------------------------------------------------------------------------
# ArrayTree — immutable Stage-3 pytree
# ---------------------------------------------------------------------------


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
    def blueprint(cls, shape: ShapeInput = ()) -> _BlueprintBase[Self]:
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
                init_kw[f.name] = val._reindex(
                    idx, new_prefix + new_shape, val.shape, get_kw
                )
            else:
                init_kw[f.name] = val

        return type(self)(**init_kw)

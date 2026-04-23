from __future__ import annotations

import dataclasses as dc
import typing
from typing import Any, Callable, Self

import equinox as eqx
import jax
import jax.numpy as jnp

ShapeType = tuple[int, ...]
InitFn = Callable[[ShapeType, Any], jax.Array]


# ---------------------------------------------------------------------------
# LeafSpec — intermediate array placeholder (Stage 2)
# ---------------------------------------------------------------------------


@dc.dataclass(frozen=True)
class LeafSpec:
    """Shape/dtype specification for an array leaf in a prototype tree.

    Intentionally NOT registered as a JAX pytree so that jax.tree.map
    treats instances as opaque leaves rather than recursing into them.
    This lets Stage-2 prototype trees be valid equinox modules.
    """

    shape: ShapeType
    dtype: Any = float


# ---------------------------------------------------------------------------
# Field helpers
# ---------------------------------------------------------------------------


def leaf(shape: ShapeType = (), dtype: Any = float, **kwargs) -> dc.Field:
    """Declare an array leaf field with shape and dtype.

    Wraps dataclasses.field(), storing a LeafSpec in the field metadata so
    that ArrayTree.default() and ArrayTree.build() can find it.

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
    """Declare a child node field and specify its shape for default construction.

    Mirrors ``leaf()`` but for ``ArrayTree`` children rather than array leaves.
    The *shape* stored here is used by ``ArrayTree.default()`` when constructing
    the child node, so the node class itself needs no hard-coded shape::

        class Vel(ArrayTree):
            vx: jax.Array = leaf(shape=(1,))
            vy: jax.Array = leaf(shape=(2,))

        class World(ArrayTree):
            vel: Vel = node(shape=(3,))   # Vel gets shape (3,) in World.default()

        proto = World.default(shape=(2,))
        # proto.vel.shape == (3,), proto.shape == (2,)
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
    dimension.  ``Ellipsis`` is counted as one slot; callers that need exact
    accounting for Ellipsis should handle it separately.
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
        """Return a new node with ``idx`` applied to all leaf arrays.

        The result's ``shape`` and ``_prefix`` are updated to reflect the
        consumed prefix dimensions.
        """
        full_prefix = self._node._prefix + self._node.shape
        remaining = full_prefix[_count_index_dims(self._idx) :]
        return self._node._reindex(self._idx, new_prefix=(), new_shape=remaining)

    def set(self, values: Any) -> ArrayTree:
        """Return a copy of the node with indexed leaves updated from *values*.

        *values* is broadcast to the structure of the node via
        ``jax.tree.broadcast``, so it can be:

        * A scalar or ``jax.Array`` — broadcast to every leaf (analogous to
          ``jnp.array(...).at[:].set(0)``).
        * An ``ArrayTree`` of the same type — each leaf is set from the
          corresponding leaf in *values*.
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
# ArrayTree
# ---------------------------------------------------------------------------


class ArrayTree(eqx.Module):
    """Base class for nested array structures built from dataclass nodes.

    Construction happens in three stages:

    **Stage 1 – Definition**
    Subclass ArrayTree and declare array leaves with ``leaf()`` and child
    nodes with ``node(shape=...)``.  Every node's nominal shape is ``()``
    unless given by its parent — no per-class shape override is needed::

        class Vel(ArrayTree):
            vx: jax.Array = leaf(shape=(1,))
            vy: jax.Array = leaf(shape=(2,))

        class World(ArrayTree):
            vel: Vel = node(shape=(3,))   # Vel gets shape (3,) when World is built

    **Stage 2 – Assembly**
    Call ``cls.default(shape=...)`` to get a prototype tree where every array
    leaf is a ``LeafSpec`` (not a real array).  The root's shape is given to
    ``default()``; each child node's shape comes from its ``node()``
    declaration.  The prototype can be inspected and modified::

        proto = World.default(shape=(2,))
        proto = eqx.tree_at(lambda t: t.vel, proto, proto.vel.with_shape((5,)))

    **Stage 3 – Instantiation**
    Call ``.build()`` (or the ``.zeros()`` / ``.ones()`` shortcuts) to
    replace every ``LeafSpec`` with a real ``jax.Array``.  The full shape
    of each leaf is the concatenation of all ancestor ``shape`` attributes
    followed by the leaf's own shape::

        world = proto.zeros()
        world.vel.vx.shape  # (2, 5, 1)  — World(2) + Vel(5) + leaf(1)

    **Indexing**
    Use ``.at[idx].get()`` / ``.at[idx].set(values)`` to index into the
    accumulated prefix of a node.  Indices are validated so they cannot
    reach into the leaf-specific shape dimensions::

        world.at[0].get()          # ok  — World has a 1-dim prefix
        world.at[0, 1].get()       # IndexError — too many dims for World
        world.vel.at[0, 3].get()   # ok  — vel's accumulated prefix is (2, 5)
        world.vel.at[0, 3, 0].get()  # IndexError
    """

    shape: ShapeType = eqx.field(static=True, kw_only=True, default=())
    # Accumulated prefix from all ancestors, set by build(). () in prototypes.
    _prefix: ShapeType = eqx.field(static=True, kw_only=True, default=(), repr=False)

    # ------------------------------------------------------------------
    # Indexing entry point
    # ------------------------------------------------------------------

    @property
    def at(self) -> _IndexHelper:
        """Entry point for prefix-validated indexing.

        Returns an ``_IndexHelper`` whose ``__getitem__`` validates the index
        against the node's accumulated prefix (``_prefix + shape``) and
        returns an ``_IndexedHelper`` with ``.get()`` and ``.set()`` methods.
        """
        return _IndexHelper(self)

    # ------------------------------------------------------------------
    # Stage 2 — prototype construction
    # ------------------------------------------------------------------

    @classmethod
    def default(cls, shape: ShapeType = ()) -> Self:
        """Build a prototype tree (Stage 2).

        Args:
            shape: The shape for this node.  Child node shapes come from
                   their ``node(shape=...)`` field declarations.

        * ``shape`` is used for this node; ``_prefix`` is left at ``()``.
        * Fields declared with ``leaf()`` become ``LeafSpec`` objects.
        * Fields declared with ``node(shape=s)`` are auto-constructed by
          calling ``ChildClass.default(shape=s)``.
        * Fields with an explicit default/factory use that instead.

        Raises ``ValueError`` if a required field has no default and its
        type cannot be automatically constructed.
        """
        kwargs: dict[str, Any] = {}
        hints = typing.get_type_hints(cls)

        for f in dc.fields(cls):
            if not f.init:
                continue

            # This node's own shape — set from argument, not the field default
            if f.name == "shape":
                kwargs["shape"] = shape
                continue

            # Other static fields (_prefix, …) — use declared default
            if f.metadata.get("static", False):
                kwargs[f.name] = _field_default(f)
                continue

            # Array leaf declared with leaf()
            if "leaf_spec" in f.metadata:
                kwargs[f.name] = f.metadata["leaf_spec"]
                continue

            # Child node: type hint must resolve to an ArrayTree subclass
            hint = hints.get(f.name)
            origin: Any = typing.get_origin(hint) or hint
            if isinstance(origin, type) and issubclass(origin, ArrayTree):
                explicit = _field_default(f, throw=False)
                if explicit is not None:
                    kwargs[f.name] = explicit
                else:
                    child_shape = f.metadata.get("node_shape", ())
                    kwargs[f.name] = origin.default(shape=child_shape)
            else:
                kwargs[f.name] = _field_default(f)

        return cls(**kwargs)

    # ------------------------------------------------------------------
    # Stage 3 — instantiation
    # ------------------------------------------------------------------

    def build(self, prefix: ShapeType = (), init_fn: InitFn = jnp.zeros) -> Self:
        """Instantiate arrays from a prototype tree (Stage 3).

        Recursively walks the tree, accumulating ``prefix + node.shape``
        at each level.  Every ``LeafSpec`` is replaced by a real
        ``jax.Array`` with shape ``accumulated + leaf.shape``, created by
        calling ``init_fn(shape, dtype)``.

        Also records each node's incoming ``prefix`` in the ``_prefix``
        field so that ``.at`` indexing can validate against the full
        accumulated prefix.

        Args:
            prefix: Extra leading dimensions prepended before this node's
                    own ``shape`` (useful for adding an outer batch axis).
            init_fn: ``(shape, dtype) -> jax.Array``.  Defaults to
                     ``jnp.zeros``.

        Returns:
            A new instance of the same type with real ``jax.Array`` leaves.
        """
        accumulated = prefix + self.shape
        kwargs: dict[str, Any] = {}

        for f in dc.fields(self):
            if not f.init:
                continue
            if f.name == "_prefix":
                kwargs["_prefix"] = prefix  # record incoming prefix
                continue
            if f.metadata.get("static", False):
                kwargs[f.name] = getattr(self, f.name)
                continue

            val = getattr(self, f.name)
            if isinstance(val, LeafSpec):
                kwargs[f.name] = init_fn(accumulated + val.shape, val.dtype)
            elif isinstance(val, ArrayTree):
                kwargs[f.name] = val.build(prefix=accumulated, init_fn=init_fn)
            else:
                kwargs[f.name] = val

        return type(self)(**kwargs)

    def zeros(self, prefix: ShapeType = ()) -> Self:
        """Build from prototype, initialising all arrays to zero."""
        return self.build(prefix=prefix, init_fn=jnp.zeros)

    def ones(self, prefix: ShapeType = ()) -> Self:
        """Build from prototype, initialising all arrays to one."""
        return self.build(prefix=prefix, init_fn=jnp.ones)

    # ------------------------------------------------------------------
    # Utilities on instantiated trees (all leaves must be jax.Array)
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

    def with_shape(self, shape: ShapeType) -> Self:
        """Return a copy of this node with a different shape (Stage 2 helper).

        Since ``shape`` is a static field it cannot be targeted by
        ``eqx.tree_at`` directly, but targeting the *node* that owns it works
        because ``eqx.tree_at`` replaces subtrees structurally::

            proto = World.default()
            proto = eqx.tree_at(lambda t: t.vel, proto, proto.vel.with_shape((5,)))
        """
        kwargs = {
            f.name: (shape if f.name == "shape" else getattr(self, f.name))
            for f in dc.fields(self)
            if f.init
        }
        return type(self)(**kwargs)

    # ------------------------------------------------------------------
    # Private helpers for .at indexing
    # ------------------------------------------------------------------

    def _reindex(self, idx: tuple, new_prefix: ShapeType, new_shape: ShapeType) -> Self:
        """Apply *idx* to all leaf arrays and update prefix/shape metadata.

        For each child ``ArrayTree``, propagates ``new_prefix + new_shape``
        as their new ``_prefix`` (reflecting what the parent now contributes)
        while leaving the child's own ``shape`` unchanged.
        """
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
    # Stage 2: build prototype — World's shape given here, Vel's shape from node()
    proto = World.default(shape=(2,))
    print("Stage 2 – prototype:")
    print(proto)

    # Optional Stage 2 edit: override a child's shape
    proto = eqx.tree_at(lambda t: t.vel, proto, proto.vel.with_shape((4,)))

    # Stage 3: instantiate
    world = proto.zeros()
    print("\nStage 3 – instantiated (zeros):")
    print(world)
    print("vel.vx shape:", world.vel.vx.shape)  # (2, 4, 1)
    print("vel.vy shape:", world.vel.vy.shape)  # (2, 4, 2)

    # Indexing
    print("\nIndexing:")
    s0 = world.at[0].get()
    print("world.at[0].get() — vel.vx shape:", s0.vel.vx.shape)  # (4, 1)
    s03 = world.vel.at[0, 3].get()
    print("world.vel.at[0,3].get() — vx shape:", s03.vx.shape)  # (1,)
    print("world.vel.at[0,3].get() — vy shape:", s03.vy.shape)  # (2,)

    # set: broadcast scalar 1.0 into world[0]
    world2 = world.at[0].set(1.0)
    print("\nworld.at[0].set(1.0) — vel.vx[0,0,0]:", world2.vel.vx[0, 0, 0])  # 1.0
    print("world.at[0].set(1.0) — vel.vx[1,0,0]:", world2.vel.vx[1, 0, 0])  # 0.0


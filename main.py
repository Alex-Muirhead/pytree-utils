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

    def set(self, values: ArrayTree) -> ArrayTree:
        """Return a copy of the node with indexed leaves updated from *values*.

        *values* should be the same type as the node returned by ``.get()``
        (i.e. same type, leaf shapes matching ``arr[idx].shape``).
        """
        return self._node._set_indexed(self._idx, values)


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
    Subclass ArrayTree and declare fields with ``leaf()`` for array leaves
    and plain type annotations for child nodes::

        class Vel(ArrayTree):
            shape: ShapeType = eqx.field(static=True, kw_only=True, default=(3,))
            vx: jax.Array = leaf(shape=(1,))
            vy: jax.Array = leaf(shape=(2,))

        class World(ArrayTree):
            shape: ShapeType = eqx.field(static=True, kw_only=True, default=(2,))
            vel: Vel

    **Stage 2 – Assembly**
    Call ``cls.default()`` to get a prototype tree where every array leaf is
    a ``LeafSpec`` (not a real array).  The prototype is a fully valid
    equinox module that can be inspected and modified::

        proto = World.default()
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
    def default(cls) -> Self:
        """Build a prototype tree (Stage 2).

        * Static fields (including ``shape`` and ``_prefix``) are filled with
          their declared defaults.
        * Fields declared with ``leaf()`` become ``LeafSpec`` objects.
        * Fields whose type hint resolves to an ``ArrayTree`` subclass are
          populated by recursively calling ``SubClass.default()``, unless
          the field already has an explicit default/factory.

        Raises ``ValueError`` if a required field has no default and its
        type cannot be automatically constructed.
        """
        kwargs: dict[str, Any] = {}
        hints = typing.get_type_hints(cls)

        for f in dc.fields(cls):
            if not f.init:
                continue

            # Static fields (shape, _prefix, …) — use declared default
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
                # Prefer an explicit default/factory over auto-construction
                explicit = _field_default(f, throw=False)
                kwargs[f.name] = explicit if explicit is not None else origin.default()
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

    def _set_indexed(self, idx: tuple, values: ArrayTree) -> Self:
        """Return a copy with indexed leaf arrays updated from *values*.

        Walks both ``self`` and *values* in parallel, applying
        ``arr.at[idx].set(val)`` to each corresponding leaf pair.
        """
        kwargs: dict[str, Any] = {
            f.name: getattr(self, f.name) for f in dc.fields(self) if f.init
        }

        for f in dc.fields(self):
            if not f.init or f.metadata.get("static", False):
                continue

            val = getattr(self, f.name)
            new_val = getattr(values, f.name)

            if isinstance(val, jax.Array):
                kwargs[f.name] = val.at[idx].set(new_val)
            elif isinstance(val, ArrayTree):
                kwargs[f.name] = val._set_indexed(idx, new_val)

        return type(self)(**kwargs)


# ---------------------------------------------------------------------------
# Example / smoke test
# ---------------------------------------------------------------------------


class Vel(ArrayTree):
    shape: ShapeType = eqx.field(static=True, kw_only=True, default=(3,))
    vx: jax.Array = leaf(shape=(1,))
    vy: jax.Array = leaf(shape=(2,))


class World(ArrayTree):
    shape: ShapeType = eqx.field(static=True, kw_only=True, default=(2,))
    vel: Vel


if __name__ == "__main__":
    # Stage 2: build prototype
    proto = World.default()
    print("Stage 2 – prototype:")
    print(proto)

    # Optional Stage 2 edit: swap a child node with a different shape.
    proto: World = eqx.tree_at(lambda t: t.vel, proto, proto.vel.with_shape((4,)))

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

    # set: put ones into world[0]
    world2 = world.at[0].set(world.at[0].get().ones_like())
    print("\nworld.at[0].set(...) — vel.vx[0,0,0]:", world2.vel.vx[0, 0, 0])  # 1.0
    print("world.at[0].set(...) — vel.vx[1,0,0]:", world2.vel.vx[1, 0, 0])  # 0.0

    test = jnp.array([1, 2, 3])
    first = test.at[:].set(0)
    print(first)

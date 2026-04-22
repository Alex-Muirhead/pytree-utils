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
        # swap in a different child, adjust shapes, etc.
        proto = eqx.tree_at(lambda t: t.vel.shape, proto, (5,))

    **Stage 3 – Instantiation**
    Call ``.build()`` (or the ``.zeros()`` / ``.ones()`` shortcuts) to
    replace every ``LeafSpec`` with a real ``jax.Array``.  The full shape
    of each leaf is the concatenation of all ancestor ``shape`` attributes
    followed by the leaf's own shape::

        world = proto.zeros()
        world.vel.vx.shape  # (2, 5, 1)  — World(2) + Vel(5) + leaf(1)
        world.vel.vy.shape  # (2, 5, 2)
    """

    shape: ShapeType = eqx.field(static=True, kw_only=True, default=())

    # ------------------------------------------------------------------
    # Stage 2 — prototype construction
    # ------------------------------------------------------------------

    @classmethod
    def default(cls) -> Self:
        """Build a prototype tree (Stage 2).

        * Static fields (including ``shape``) are filled with their declared
          defaults.
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

            # Static fields (shape, etc.) — use declared default
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
    proto = eqx.tree_at(lambda t: t.vel, proto, proto.vel.with_shape((4,)))

    # Stage 3: instantiate
    world = proto.zeros()
    print("\nStage 3 – instantiated (zeros):")
    print(world)
    print("vel.vx shape:", world.vel.vx.shape)  # (2, 4, 1)
    print("vel.vy shape:", world.vel.vy.shape)  # (2, 4, 2)

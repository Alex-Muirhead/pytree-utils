"""Leaf/node field-spec types and field-declaration helpers."""

from __future__ import annotations

import dataclasses as dc
from typing import Any, Protocol

import jax

ShapeType = tuple[int, ...]
ShapeInput = int | ShapeType


class InitFn(Protocol):
    """Callable signature for the per-leaf array initialiser.

    ``dtype`` is keyword-only so a pre-bound ``functools.partial(jnp.full,
    fill_value=...)`` -- which only exposes ``dtype`` by keyword -- still
    satisfies the protocol.
    """

    def __call__(self, shape: ShapeType, *, dtype: Any = ...) -> jax.Array: ...


def to_shape(s: ShapeInput) -> ShapeType:
    """Normalise a shape input, wrapping a bare ``int`` into a 1-tuple."""
    return (s,) if isinstance(s, int) else s


@dc.dataclass(frozen=True)
class LeafSpec:
    """Shape/dtype specification for an array leaf.

    Lives inside a blueprint and is replaced by a real ``jax.Array`` when the
    blueprint is built.
    """

    shape: ShapeType
    dtype: Any = float


def leaf(shape: ShapeInput = (), dtype: Any = float, **kwargs) -> Any:
    """Declare an array leaf field with shape and dtype::

        class MyNode(ArrayTree):
            x: jax.Array = leaf(shape=(3,))
            y: jax.Array = leaf(shape=(4,), dtype=jnp.float16)

    Returns ``Any`` (rather than ``dc.Field``) so the call can be assigned to a
    typed field annotation, matching ``dataclasses.field``'s own convention.
    """
    metadata = dict(kwargs.pop("metadata", None) or {})
    if "leaf_spec" in metadata:
        raise ValueError("leaf_spec multiply defined in metadata")
    metadata["leaf_spec"] = LeafSpec(shape=to_shape(shape), dtype=dtype)
    return dc.field(**kwargs, metadata=metadata)


def node(shape: ShapeInput = (), **kwargs) -> Any:
    """Declare a child node field and its shape.

    The *shape* is applied when the parent's ``Blueprint`` is constructed, so
    the node class itself needs no hard-coded shape::

        class Vel(ArrayTree):
            vx: jax.Array = leaf(shape=(1,))

        class World(ArrayTree):
            vel: Vel = node(shape=(3,))

        proto = blueprint(World, shape=(2,))
        # proto.shape == (2,), proto.vel.shape == (3,)
    """
    metadata = dict(kwargs.pop("metadata", None) or {})
    if "node_shape" in metadata:
        raise ValueError("node_shape multiply defined in metadata")
    metadata["node_shape"] = to_shape(shape)
    return dc.field(**kwargs, metadata=metadata)


def field_default(field: dc.Field, *, throw: bool = True) -> Any:
    """Return a dataclass field's default value (from its default or factory)."""
    if field.default is not dc.MISSING:
        return field.default
    if field.default_factory is not dc.MISSING:
        return field.default_factory()
    if throw:
        raise ValueError(f"Field '{field.name}' has no default value or factory")
    return None


def count_index_dims(idx: tuple) -> int:
    """Count how many existing dimensions an index tuple addresses.

    Every element other than ``None`` / ``numpy.newaxis`` addresses one
    dimension.
    """
    return sum(i is not None for i in idx)

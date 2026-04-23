"""Leaf/node field-spec types and Stage-1 field-declaration helpers."""

from __future__ import annotations

import dataclasses as dc
from collections.abc import Callable
from typing import Any

import jax

ShapeType = tuple[int, ...]
InitFn = Callable[[ShapeType, Any], jax.Array]


@dc.dataclass(frozen=True)
class LeafSpec:
    """Shape/dtype specification for an array leaf.

    Lives inside a ``Blueprint`` (Stage 2) and is replaced by a real
    ``jax.Array`` when the Blueprint is built (Stage 3).
    """

    shape: ShapeType
    dtype: Any = float


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

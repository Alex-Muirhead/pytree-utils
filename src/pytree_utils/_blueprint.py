"""Blueprint machinery for ArrayTree."""

import dataclasses as dc
import functools
import typing
from typing import Any, ClassVar

import jax.numpy as jnp

from pytree_utils._spec import (
    InitFn,
    LeafSpec,
    ShapeInput,
    ShapeType,
    _field_default,
    _to_shape,
)


class _BlueprintBase:
    """Base for all generated Blueprint types.

    Blueprints are plain mutable Python objects (not JAX pytrees). They
    describe the shape/dtype structure of an ``ArrayTree`` before any arrays
    are allocated. Mutate fields directly, then call ``.zeros()`` or
    ``.ones()`` to produce a fully instantiated ``ArrayTree``.
    """

    __slots__ = ()

    _array_tree_cls: ClassVar[type]

    def _build(self, prefix: ShapeInput = (), init_fn: InitFn = jnp.zeros) -> Any:
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

    def zeros(self, prefix: ShapeInput = ()) -> Any:
        """Build with zero-filled arrays."""
        return self._build(prefix=prefix, init_fn=jnp.zeros)

    def ones(self, prefix: ShapeInput = ()) -> Any:
        """Build with one-filled arrays."""
        return self._build(prefix=prefix, init_fn=jnp.ones)

    def full(self, fill_value: Any, prefix: ShapeInput = ()) -> Any:
        """Build with value-filled arrays."""
        init_fn = functools.partial(jnp.full, fill_value=fill_value)
        return self._build(prefix=prefix, init_fn=init_fn)

    def empty(self, prefix: ShapeInput = ()) -> Any:
        """Build with empty arrays."""
        return self._build(prefix=prefix, init_fn=jnp.empty)


def blueprint(cls_or_alias: Any, shape: ShapeInput = ()) -> _BlueprintBase:
    """Create a mutable Blueprint for an ArrayTree subclass.

    Accepts either a concrete ``ArrayTree`` subclass or a generic alias
    such as ``Container[Vel]``::

        bp1 = blueprint(World, shape=(2,))
        bp2 = blueprint(Container[Vel], shape=(2,))
    """
    cls, type_map = _split_alias(cls_or_alias)
    bp_cls = _get_blueprint_cls(cls, type_map)
    return bp_cls(shape=_to_shape(shape))


def _split_alias(cls_or_alias: Any) -> tuple[type, dict]:
    """Return ``(array_tree_cls, type_map)`` for a class or generic alias."""
    if isinstance(cls_or_alias, type):
        return cls_or_alias, {}
    origin = typing.get_origin(cls_or_alias)
    args = typing.get_args(cls_or_alias)
    if origin is None or not args:
        raise TypeError(f"Cannot create blueprint from {cls_or_alias!r}")
    type_params = getattr(origin, "__type_params__", ())
    return origin, dict(zip(type_params, args, strict=False))


def _get_blueprint_cls(array_tree_cls: type, type_map: dict) -> type:
    """Return the Blueprint class for an ArrayTree subclass.

    For the unparameterized case the result is cached on the class. For a
    non-empty *type_map* a fresh class is built each call -- parameterized
    blueprint classes are cheap to build and would otherwise need a
    composite cache key.
    """
    if type_map:
        return _make_blueprint_cls(array_tree_cls, type_map)
    if "_blueprint_cls" not in array_tree_cls.__dict__:
        array_tree_cls._blueprint_cls = _make_blueprint_cls(array_tree_cls, {})  # type: ignore[attr-defined]
    return array_tree_cls._blueprint_cls  # type: ignore[attr-defined]


def _make_blueprint_cls(array_tree_cls: type, type_map: dict) -> type:
    """Dynamically generate a mutable Blueprint dataclass for *array_tree_cls*.

    *type_map* maps ``TypeVar`` objects to concrete ``ArrayTree`` subclasses,
    used when the class is generic (e.g. ``Container[Vel]``).
    """
    from pytree_utils.array_tree import ArrayTree  # lazy import to avoid circular dep

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

        child_cls, child_type_map = _resolve_child(hints.get(f.name), type_map)
        if child_cls is None or not issubclass(child_cls, ArrayTree):
            continue

        node_shape = f.metadata.get("node_shape", ())
        child_bp_cls = _get_blueprint_cls(child_cls, child_type_map)
        factory = functools.partial(child_bp_cls, shape=node_shape)
        fields.append((f.name, child_bp_cls, dc.field(default_factory=factory)))

    blueprint_cls = dc.make_dataclass(
        f"{array_tree_cls.__name__}Blueprint",
        fields,
        bases=(_BlueprintBase,),
        slots=True,
    )
    blueprint_cls._array_tree_cls = array_tree_cls  # type: ignore[attr-defined]
    return blueprint_cls


def _resolve_child(hint: Any, type_map: dict) -> tuple[type | None, dict]:
    """Reduce a field annotation to ``(child_cls, child_type_map)``.

    Substitutes any ``TypeVar`` in *hint* (or its generic args) using
    *type_map*. Returns ``(None, {})`` when the hint is not an ArrayTree-like
    class or alias.
    """
    if isinstance(hint, typing.TypeVar):
        hint = type_map.get(hint, hint)
    if isinstance(hint, type):
        return hint, {}
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin is None or not isinstance(origin, type):
        return None, {}
    resolved = tuple(
        type_map.get(a, a) if isinstance(a, typing.TypeVar) else a for a in args
    )
    type_params = getattr(origin, "__type_params__", ())
    return origin, dict(zip(type_params, resolved, strict=False))

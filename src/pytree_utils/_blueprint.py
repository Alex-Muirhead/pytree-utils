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

    def __class_getitem__(cls, item: Any) -> type:
        return cls

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


@dc.dataclass(frozen=True)
class _ParameterizedTree:
    """Returned by ``GenericArrayTree[ConcreteType]``; provides ``.blueprint()``."""

    cls: type
    type_map: dict

    def blueprint(self, shape: ShapeInput = ()) -> _BlueprintBase:
        """Create a Blueprint with TypeVars resolved to their concrete types."""
        return _make_blueprint_cls(self.cls, self.type_map)(shape=_to_shape(shape))


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

        hint = _resolve_hint(hints.get(f.name), type_map)
        node_shape = f.metadata.get("node_shape", ())

        if isinstance(hint, _ParameterizedTree):
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

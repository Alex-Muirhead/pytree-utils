"""Blueprint machinery for ArrayTree."""

import dataclasses as dc
import functools
import typing
from copy import copy
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from pytree_utils._spec import (
    InitFn,
    LeafSpec,
    ShapeInput,
    ShapeType,
    _field_default,
    _to_shape,
)
from pytree_utils.array_tree import ArrayTree

type TypeMap = dict[typing.TypeVar, Any]


class BlueprintBase[T: ArrayTree]:
    """Base for all generated Blueprint types.

    Blueprints are plain mutable Python objects (not JAX pytrees). They
    describe the shape/dtype structure of an ``ArrayTree`` before any arrays
    are allocated. Mutate fields directly, then call ``.zeros()`` or
    ``.ones()`` to produce a fully instantiated ``ArrayTree``.

    *T* is the target ``ArrayTree`` subclass that ``.zeros()`` / ``.ones()``
    / ``.full()`` / ``.empty()`` return.
    """

    # Load-bearing: Python only enforces ``__slots__`` end-to-end when every
    # ancestor declares slots. Generated subclasses use ``slots=True`` to
    # prevent typo'd field assignments, which only works if this base also
    # opts out of ``__dict__``.
    __slots__ = ()

    shape: ShapeType
    _array_tree_cls: type[T]

    if TYPE_CHECKING:
        # Concrete blueprint subclasses are generated dynamically from each
        # ArrayTree subclass's field list, so their fields are invisible to
        # static type checkers. These stubs keep ``proto.vel`` /
        # ``proto.child`` accesses typed as Any, and the ``cls(shape=...)``
        # construction site type-checks without our needing a full
        # dataclass_transform.  At runtime, the generated dataclass supplies
        # the real ``__init__`` and ``__slots__`` enforces field discipline.
        # def __init__(self, *, shape: ShapeType = (), **kwargs: Any) -> None: ...
        def __getattr__(self, name: str) -> Any: ...

    def _build(self, prefix: ShapeInput = (), init_fn: InitFn = jnp.zeros) -> T:
        """Instantiate arrays from this blueprint (Stage 3).

        Args:
            prefix: Extra leading dimensions prepended outside this node's
                    own ``shape``.
            init_fn: ``(shape, dtype=...) -> jax.Array``.  Defaults to
                     ``jnp.zeros``.
        """
        cls = self._array_tree_cls
        prefix = _to_shape(prefix)
        accumulated = prefix + self.shape
        # Init-kwargs assemble heterogeneous values (jax.Arrays, sub-ArrayTrees,
        # ShapeTypes, static-field defaults), so the value side is Any. The
        # dedicated ``_shape`` leaf carries the node's accumulated prefix with a
        # zero-sized trailing axis, so it occupies no memory yet rides along with
        # the data under indexing; ``ArrayTree.shape`` just reads it back.
        kwargs: dict[str, Any] = {"_shape": jnp.empty((*accumulated, 0))}

        for f in dc.fields(cls):
            if not f.init or f.name == "_shape":
                continue
            if f.metadata.get("static", False):
                kwargs[f.name] = _field_default(f)
                continue

            val = getattr(self, f.name)
            if isinstance(val, LeafSpec):
                kwargs[f.name] = init_fn(accumulated + val.shape, dtype=val.dtype)
            elif isinstance(val, BlueprintBase):
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


def blueprint[T: ArrayTree](cls_or_alias: type[T], shape: ShapeInput = ()) -> BlueprintBase[T]:
    """Create a mutable Blueprint for an ArrayTree subclass.

    Accepts either a concrete ``ArrayTree`` subclass or a generic alias
    such as ``Container[Vel]``::

        bp1 = blueprint(World, shape=(2,))
        bp2 = blueprint(Container[Vel], shape=(2,))
    """
    cls, type_map = get_type_mapping(cls_or_alias)
    bp_cls = make_blueprint_cls(cls, type_map)
    return bp_cls(shape=_to_shape(shape))


def make_blueprint_cls[T: ArrayTree](array_tree_cls: type[T], type_map: TypeMap) -> type[BlueprintBase[T]]:
    """Dynamically generate a mutable Blueprint dataclass for *array_tree_cls*.

    *type_map* maps ``TypeVar`` objects to concrete ``ArrayTree`` subclasses,
    used when the class is generic (e.g. ``Container[Vel]``).
    """
    # Matches the ``dc.make_dataclass`` ``fields`` signature: ``(name,
    # type-annotation, default-or-Field)``.  Both trailing slots are ``Any``
    # because the entries are heterogeneous (LeafSpec, sub-blueprint class,
    # ShapeType) and ``dc.field``'s overloaded return type collapses to the
    # default value's type, not ``dc.Field``.
    fields: list[tuple[str, Any, Any]] = [("shape", ShapeType, dc.field(default=()))]
    hints = typing.get_type_hints(array_tree_cls)

    for f in dc.fields(array_tree_cls):
        if f.name == "_shape" or not f.init:
            continue
        if f.metadata.get("static", False):
            continue

        if "leaf_spec" in f.metadata:
            fields.append((f.name, LeafSpec, dc.field(default=f.metadata["leaf_spec"])))
            continue

        child_cls, child_type_map = resolve_child(hints.get(f.name), type_map)
        if child_cls is None:
            continue

        node_shape = f.metadata.get("node_shape", ())
        child_bp_cls = make_blueprint_cls(child_cls, child_type_map)
        factory = functools.partial(child_bp_cls, shape=node_shape)
        fields.append((f.name, child_bp_cls, dc.field(default_factory=factory)))

    blueprint_cls: type[BlueprintBase[T]] = typing.cast(
        type[BlueprintBase[T]],
        dc.make_dataclass(
            array_tree_cls.__name__ + "Blueprint",
            fields,
            bases=(BlueprintBase,),
            slots=True,
            namespace={"_array_tree_cls": array_tree_cls},
        ),
    )
    return blueprint_cls


def get_type_mapping(typ_: type) -> tuple[type, TypeMap]:
    """Return ``(array_tree_cls, type_map)`` for a class or generic alias."""
    origin = typing.get_origin(typ_)
    if origin is None:
        # Not a generic
        return typ_, {}
    args = typing.get_args(typ_)
    if not args:
        # Not a typed-generic
        return origin, {}
    type_params: tuple[typing.TypeVar, ...] = getattr(origin, "__type_params__", ())
    return origin, dict(zip(type_params, args, strict=False))


def resolve_child(hint: Any, type_map: TypeMap) -> tuple[type[ArrayTree] | None, TypeMap]:
    """Reduce a field annotation to ``(child_cls, child_type_map)``.

    Substitutes any ``TypeVar`` in *hint* (or its generic args) using
    *type_map*. Returns ``(None, {})`` when the hint is not an ArrayTree
    subclass or alias.
    """
    if isinstance(hint, typing.TypeVar):
        hint = type_map.get(hint, hint)
    if isinstance(hint, type):
        return (hint, {}) if issubclass(hint, ArrayTree) else (None, {})
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin is None or not isinstance(origin, type) or not issubclass(origin, ArrayTree):
        return None, {}
    resolved = tuple(type_map.get(a, a) if isinstance(a, typing.TypeVar) else a for a in args)
    type_params: tuple[typing.TypeVar, ...] = getattr(origin, "__type_params__", ())
    return origin, dict(zip(type_params, resolved, strict=False))


# New stuff
#
# typespace is like "namespace" but for types


def get_typespace(hint, typespace: dict | None = None):
    typespace = typespace or {}
    origin = typing.get_origin(hint) or hint
    generic_args = typing.get_args(hint)
    concrete_args = (typespace.get(arg, arg) for arg in generic_args)
    return dict(zip(origin.__type_params__, concrete_args, strict=True))


def realise_generics(hint, typespace: dict | None = None):
    typespace = typespace or {}
    if isinstance(hint, typing.TypeVar):
        # Should we throw an error, since we didn't realise it?
        return typespace.get(hint, hint)
    origin = typing.get_origin(hint)
    if origin is None:
        return hint
    generic_args = typing.get_args(hint)
    concrete_args = (typespace.get(arg, arg) for arg in generic_args)
    return typing.GenericAlias(origin, concrete_args)  # ty:ignore[unresolved-attribute]


def generic_fields(dataclass, typespace: dict | None = None):
    typespace = typespace or {}
    origin = typing.get_origin(dataclass) or dataclass
    generic_fields = []
    local_typespace = get_typespace(dataclass, typespace)
    for field in map(copy, dc.fields(origin)):
        # Need to copy to avoid modifying the original
        field.type = realise_generics(field.type, local_typespace)
        generic_fields.append(field)
    return tuple(generic_fields)

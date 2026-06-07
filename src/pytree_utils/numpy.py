import functools

import jax
import jax.numpy as jnp
from jaxtyping import PyTree


def zeros_like[T: PyTree](tree: T) -> T:
    """Return a copy with all arrays replaced by zeros of the same shape."""
    return jax.tree.map(jnp.zeros_like, tree)


def ones_like[T: PyTree](tree: T) -> T:
    """Return a copy with all arrays replaced by ones of the same shape."""
    return jax.tree.map(jnp.ones_like, tree)


def empty_like[T: PyTree](tree: T) -> T:
    """Return a copy with all arrays replaced by uninitialised arrays."""
    return jax.tree.map(jnp.empty_like, tree)


def full_like[T: PyTree](tree: T, fill_value: object) -> T:
    """Return a copy with all arrays replaced by fill_value of the same shape."""
    init_fn = functools.partial(jnp.full_like, fill_value=fill_value)
    return jax.tree.map(init_fn, tree)

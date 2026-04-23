# pytree-utils

A small helper library for working with nested structs-of-arrays in JAX,
built on top of [Equinox](https://github.com/patrick-kidger/equinox).

The core idea is a three-stage workflow:

1. **Define** your structure using plain dataclass notation.
2. **Configure** a mutable *Blueprint* before allocating any memory.
3. **Instantiate** an immutable JAX pytree from the blueprint.

## Installation

```bash
pip install pytree-utils
```

Requires Python 3.12+ and JAX.

## Basic example

```python
import jax.numpy as jnp
from pytree_utils import ArrayTree, leaf, node

class Velocity(ArrayTree):
    vx: jax.Array = leaf(shape=1)
    vy: jax.Array = leaf(shape=2)

class World(ArrayTree):
    vel: Velocity = node(shape=3)

# Stage 2 - mutable blueprint
proto = World.blueprint(shape=2)
proto.vel.shape = 4          # override the node shape before allocating

# Stage 3 - allocate
world = proto.zeros()

world.vel.vx.shape  # (2, 4, 1) -- World(2) x Vel(4) x leaf(1)
world.vel.vy.shape  # (2, 4, 2)
```

Leaf array shapes are the concatenation of all ancestor shapes plus the
leaf's own shape. The blueprint lets you adjust those shapes freely before
committing to an allocation.

## Indexing

`.at[idx]` restricts indexing to the *prefix* dimensions (the accumulated
ancestor shapes), so you can't accidentally slice into a leaf's own axes.

```python
world.at[0].get()           # shape (4, 1) -- drops the World(2) prefix
world.vel.at[0, 3].get()    # shape (1,)  -- drops World(2) x Vel(4)
world.at[0].set(1.0)        # broadcast scalar into all leaves at index 0
```

## Arithmetic

Standard arithmetic and comparisons work elementwise across the tree,
accepting either a scalar or a same-structure `ArrayTree` on the right-hand
side.

```python
a = World.blueprint(shape=2).ones()
b = a * 2.0        # all leaves scaled
c = a + b          # elementwise, leaf by leaf
mask = a < b       # returns a same-structure tree of booleans
```

## Generic nodes

If a node type is generic over its child, you can parameterise it at
blueprint time.

```python
class Container[T: ArrayTree](ArrayTree):
    child: T = node(shape=5)

bp = Container[Velocity].blueprint(shape=2)
w  = bp.zeros()
w.child.vx.shape  # (2, 5, 1) -- Container(2) x node(5) x leaf(1)
```

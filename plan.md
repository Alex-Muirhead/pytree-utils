## Blueprint stuff

```python
class Foo(ArrayTree):
  x: jax.Array  # implicitly -> `= leaf()`
  y: jax.Array = leaf(shape=3)
  z: jax.Array = leaf(shape=(2,2), dtype=int)


bp = blueprint(Foo)
foo = bp.zeros(shape=3)  # Prefixes the shape
foo.shape  # == 3

class Bar(ArrayTree):
  a: jax.Array  # implicitly -> `= leaf()`
  b: Foo  # implicitly, due to type being ArrayTree -> `= node()`

bar = blueprint(Bar).ones()
    
```

## Array stuff

```python
class Vel(ArrayTree):
    vx: jax.Array = leaf(shape=(1,))
    vy: jax.Array = leaf(shape=(2,))


class World(ArrayTree):
    vel: Vel = node(shape=(3,))


bp = blueprint(World, shape=2)
world = bp.zeros()
world.shape  # == (2,)
world.vel.shape  # == (2, 3)
```

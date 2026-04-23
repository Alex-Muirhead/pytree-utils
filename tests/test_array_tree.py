import pytest
import jax
import jax.numpy as jnp

from pytree_utils import ArrayTree, leaf, node


class Vel(ArrayTree):
    vx: jax.Array = leaf(shape=(1,))
    vy: jax.Array = leaf(shape=(2,))


class World(ArrayTree):
    vel: Vel = node(shape=(3,))


@pytest.fixture
def world():
    proto = World.blueprint(shape=(2,))
    proto.vel.shape = (4,)
    return proto.zeros()


def test_leaf_shapes(world):
    assert world.vel.vx.shape == (2, 4, 1)
    assert world.vel.vy.shape == (2, 4, 2)


def test_at_get_root(world):
    s = world.at[0].get()
    assert s.vel.vx.shape == (4, 1)
    assert s.vel.vy.shape == (4, 2)


def test_at_get_child(world):
    s = world.vel.at[0, 3].get()
    assert s.vx.shape == (1,)
    assert s.vy.shape == (2,)


def test_at_get_too_many_dims(world):
    with pytest.raises(IndexError):
        world.at[0, 1].get()


def test_at_set_scalar(world):
    world2 = world.at[0].set(1.0)
    assert float(world2.vel.vx[0, 0, 0]) == 1.0
    assert float(world2.vel.vx[1, 0, 0]) == 0.0


def test_blueprint_mutation():
    proto = World.blueprint(shape=(2,))
    proto.vel.shape = (5,)
    w = proto.zeros()
    assert w.vel.vx.shape == (2, 5, 1)


def test_blueprint_slots():
    proto = World.blueprint(shape=(2,))
    with pytest.raises(AttributeError):
        proto.nonexistent = 42


def test_zeros_like_ones_like(world):
    z = world.zeros_like()
    assert float(z.vel.vx[0, 0, 0]) == 0.0
    o = world.ones_like()
    assert float(o.vel.vx[0, 0, 0]) == 1.0

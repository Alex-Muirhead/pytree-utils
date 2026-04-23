import pytest
import jax

from pytree_utils import ArrayTree, leaf, node


class Vel(ArrayTree):
    vx: jax.Array = leaf(shape=(1,))
    vy: jax.Array = leaf(shape=(2,))


class World(ArrayTree):
    vel: Vel = node(shape=(3,))


@pytest.fixture
def world() -> World:
    proto = World.blueprint(shape=(2,))
    proto.vel.shape = (4,)
    return proto.zeros()


def test_leaf_shapes(world: World):
    assert world.vel.vx.shape == (2, 4, 1)
    assert world.vel.vy.shape == (2, 4, 2)


def test_at_get_root(world: World):
    s = world.at[0].get()
    assert s.vel.vx.shape == (4, 1)
    assert s.vel.vy.shape == (4, 2)


def test_at_get_child(world: World):
    s = world.vel.at[0, 3].get()
    assert s.vx.shape == (1,)
    assert s.vy.shape == (2,)


def test_at_get_too_many_dims(world: World):
    with pytest.raises(IndexError):
        world.at[0, 1].get()


def test_at_set_scalar(world: World):
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


def test_zeros_like_ones_like(world: World):
    z = world.zeros_like()
    assert float(z.vel.vx[0, 0, 0]) == 0.0
    o = world.ones_like()
    assert float(o.vel.vx[0, 0, 0]) == 1.0


# ---------------------------------------------------------------------------
# Generic ArrayTree tests
# ---------------------------------------------------------------------------


class Pos(ArrayTree):
    x: jax.Array = leaf(shape=(3,))


class Container[T: ArrayTree](ArrayTree):
    child: T = node(shape=(5,))


class Wrapper(ArrayTree):
    one: Container[Pos] = node(shape=(1,))
    two: Container[Vel] = node(shape=(2,))


def test_generic_blueprint_uses_concrete_type():
    bp = Container[Vel].blueprint(shape=(2,))
    # child should be a VelBlueprint with node_shape default (5,)
    from pytree_utils.array_tree import _BlueprintBase

    assert isinstance(bp.child, _BlueprintBase)
    assert bp.child.shape == (5,)


def test_generic_leaf_shapes():
    w = Container[Vel].blueprint(shape=(2,)).zeros()
    assert w.child.vx.shape == (2, 5, 1)
    assert w.child.vy.shape == (2, 5, 2)


def test_generic_different_concrete_types():
    wp = Container[Pos].blueprint(shape=(2,)).zeros()
    assert wp.child.x.shape == (2, 5, 3)


def test_generic_blueprint_mutation():
    bp = Container[Vel].blueprint(shape=(2,))
    bp.child.shape = (7,)
    w = bp.zeros()
    assert w.child.vx.shape == (2, 7, 1)


def test_shape_int_shorthand():
    class Speed(ArrayTree):
        v: jax.Array = leaf(shape=3)

    class Track(ArrayTree):
        s: Speed = node(shape=2)

    w = Track.blueprint(shape=4).zeros()
    assert w.s.v.shape == (4, 2, 3)


def test_generic_children():
    bp = Wrapper.blueprint(shape=(3,))
    w = bp.zeros()
    assert hasattr(w.one.child, "x")
    assert hasattr(w.two.child, "vx")
    assert w.one.child.x.shape == (3, 1, 5, 3)
    assert w.two.child.vx.shape == (3, 2, 5, 1)

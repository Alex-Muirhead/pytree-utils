import jax
import pytest

from pytree_utils import ArrayTree, blueprint, leaf, node


class Vel(ArrayTree):
    vx: jax.Array = leaf(shape=(1,))
    vy: jax.Array = leaf(shape=(2,))


class World(ArrayTree):
    vel: Vel = node(shape=(3,))


@pytest.fixture
def world() -> World:
    proto = blueprint(World, shape=(2,))
    proto.vel.shape = (4,)
    return proto.zeros()


def test_leaf_shapes(world: World):
    assert world.vel.vx.shape == (2, 4, 1)
    assert world.vel.vy.shape == (2, 4, 2)


def test_node_shape_is_full_accumulated(world: World):
    # A node reports its full accumulated shape, not just its own block.
    assert world.shape == (2,)
    assert world.vel.shape == (2, 4)


def test_node_shape_follows_indexing(world: World):
    # Indexing only slices the leaves; node shapes derive automatically.
    assert world.at[0].get().vel.shape == (4,)
    assert world.vel.at[0, 3].get().shape == ()


def test_leafless_node_reports_full_shape():
    # A node whose subtree holds no data still reports its accumulated shape,
    # via the dedicated zero-sized shape leaf.
    class Empty(ArrayTree):
        pass

    class Holder(ArrayTree):
        e: Empty = node(shape=(3,))

    h = blueprint(Holder, shape=(2,)).zeros()
    assert h.shape == (2,)
    assert h.e.shape == (2, 3)


def test_shape_leaf_costs_no_memory(world: World):
    # The shape markers are real leaves but hold zero elements.
    assert all(leaf.nbytes > 0 or leaf.size == 0 for leaf in jax.tree.leaves(world))
    assert any(leaf.size == 0 for leaf in jax.tree.leaves(world))


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
    proto = blueprint(World, shape=(2,))
    proto.vel.shape = (5,)
    w = proto.zeros()
    assert w.vel.vx.shape == (2, 5, 1)


def test_blueprint_slots():
    proto = blueprint(World, shape=(2,))
    with pytest.raises(AttributeError):
        proto.nonexistent = 42  # type: ignore unresolved-attribute


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
    bp = blueprint(Container[Vel], shape=(2,))
    # child should be a VelBlueprint with node_shape default (5,)
    from pytree_utils._blueprint import BlueprintBase

    assert isinstance(bp.child, BlueprintBase)
    assert bp.child.shape == (5,)


def test_generic_leaf_shapes():
    w = blueprint(Container[Vel], shape=(2,)).zeros()
    assert w.child.vx.shape == (2, 5, 1)
    assert w.child.vy.shape == (2, 5, 2)


def test_generic_different_concrete_types():
    wp = blueprint(Container[Pos], shape=(2,)).zeros()
    assert wp.child.x.shape == (2, 5, 3)


def test_generic_blueprint_mutation():
    bp = blueprint(Container[Vel], shape=(2,))
    bp.child.shape = (7,)
    w = bp.zeros()
    assert w.child.vx.shape == (2, 7, 1)


def test_shape_int_shorthand():
    class Speed(ArrayTree):
        v: jax.Array = leaf(shape=3)

    class Track(ArrayTree):
        s: Speed = node(shape=2)

    w = blueprint(Track, shape=4).zeros()
    assert w.s.v.shape == (4, 2, 3)


def test_generic_children():
    bp = blueprint(Wrapper, shape=(3,))
    w = bp.zeros()
    assert hasattr(w.one.child, "x")
    assert hasattr(w.two.child, "vx")
    assert w.one.child.x.shape == (3, 1, 5, 3)
    assert w.two.child.vx.shape == (3, 2, 5, 1)

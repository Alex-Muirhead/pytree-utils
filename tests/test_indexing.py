import jax.numpy as jnp
import pytest

from pytree_utils import ArrayTree, blueprint, leaf, node


class Vel(ArrayTree):
    vx: jnp.ndarray = leaf(shape=1)
    vy: jnp.ndarray = leaf(shape=2)


class World(ArrayTree):
    vel: Vel = node(shape=3)


@pytest.fixture
def zeros():
    return blueprint(World, shape=2).zeros()


@pytest.fixture
def ones():
    return blueprint(World, shape=2).ones()


@pytest.fixture
def threes(ones):
    return ones * 3.0


# ----------------------------------------------------------------------
# get
# ----------------------------------------------------------------------


def test_get_reduces_prefix(ones):
    s = ones.at[0].get()
    assert s.vel.vx.shape == (3, 1)


def test_get_child(ones):
    s = ones.vel.at[0, 2].get()
    assert s.vx.shape == (1,)


def test_get_fill_value(ones):
    s = ones.at[5].get(mode="fill", fill_value=0.0)
    assert float(s.vel.vx[0, 0]) == 0.0


# ----------------------------------------------------------------------
# set
# ----------------------------------------------------------------------


def test_set_scalar(zeros):
    w = zeros.at[0].set(5.0)
    assert float(w.vel.vx[0, 0, 0]) == 5.0
    assert float(w.vel.vx[1, 0, 0]) == 0.0


# ----------------------------------------------------------------------
# add
# ----------------------------------------------------------------------


def test_add(ones):
    w = ones.at[0].add(4.0)
    assert float(w.vel.vx[0, 0, 0]) == 5.0
    assert float(w.vel.vx[1, 0, 0]) == 1.0


# ----------------------------------------------------------------------
# mul
# ----------------------------------------------------------------------


def test_mul(threes):
    w = threes.at[0].mul(2.0)
    assert float(w.vel.vx[0, 0, 0]) == 6.0
    assert float(w.vel.vx[1, 0, 0]) == 3.0


# ----------------------------------------------------------------------
# min
# ----------------------------------------------------------------------


def test_min(threes):
    w = threes.at[0].min(2.0)
    assert float(w.vel.vx[0, 0, 0]) == 2.0
    assert float(w.vel.vx[1, 0, 0]) == 3.0


# ----------------------------------------------------------------------
# max
# ----------------------------------------------------------------------


def test_max(ones):
    w = ones.at[0].max(5.0)
    assert float(w.vel.vx[0, 0, 0]) == 5.0
    assert float(w.vel.vx[1, 0, 0]) == 1.0


# ----------------------------------------------------------------------
# apply
# ----------------------------------------------------------------------


def test_apply(threes):
    w = threes.at[0].apply(lambda x: x * 10)
    assert float(w.vel.vx[0, 0, 0]) == 30.0
    assert float(w.vel.vx[1, 0, 0]) == 3.0


# ----------------------------------------------------------------------
# path-scoped .at -- world.at.<field>[idx]...
# ----------------------------------------------------------------------


def test_path_set_preserves_root(ones):
    w = ones.at.vel[0].set(9.0)
    assert type(w) is World
    assert w.shape == (2,)
    assert float(w.vel.vx[0, 0, 0]) == 9.0
    assert float(w.vel.vx[1, 0, 0]) == 1.0  # sibling slice untouched


def test_path_add_preserves_root(ones):
    w = ones.at.vel[0].add(4.0)
    assert type(w) is World
    assert w.shape == (2,)
    assert float(w.vel.vx[0, 0, 0]) == 5.0
    assert float(w.vel.vx[1, 0, 0]) == 1.0


def test_path_mul_min_max(threes):
    assert float(threes.at.vel[0].mul(2.0).vel.vx[0, 0, 0]) == 6.0
    assert float(threes.at.vel[0].min(2.0).vel.vx[0, 0, 0]) == 2.0
    assert float(threes.at.vel[0].max(5.0).vel.vx[0, 0, 0]) == 5.0
    # untouched slice survives each op
    assert float(threes.at.vel[0].mul(2.0).vel.vx[1, 0, 0]) == 3.0


def test_path_apply_preserves_root(threes):
    w = threes.at.vel[0].apply(lambda x: x * 10)
    assert type(w) is World
    assert float(w.vel.vx[0, 0, 0]) == 30.0
    assert float(w.vel.vx[1, 0, 0]) == 3.0


def test_path_leaf_target(ones):
    w = ones.at.vel.vx[0].set(7.0)
    assert type(w) is World
    assert float(w.vel.vx[0, 0, 0]) == 7.0
    assert float(w.vel.vx[1, 0, 0]) == 1.0
    assert float(w.vel.vy[0, 0, 0]) == 1.0  # other leaf untouched


def test_path_get_narrows_like_node_form(ones):
    via_path = ones.at.vel[0].get()
    via_node = ones.vel.at[0].get()
    assert type(via_path) is Vel
    assert via_path.vx.shape == (3, 1)
    assert via_path.vx.shape == via_node.vx.shape


def test_path_get_leaf_returns_array(ones):
    arr = ones.at.vel.vx[0].get()
    assert arr.shape == (3, 1)


def test_path_over_index_rejected(ones):
    with pytest.raises(IndexError):
        ones.at.vel[0, 1, 2]


def test_path_bad_field_rejected(ones):
    with pytest.raises(AttributeError):
        _ = ones.at.nope

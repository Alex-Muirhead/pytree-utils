import jax.numpy as jnp
import pytest

from pytree_utils import ArrayTree, leaf, node


class Vec(ArrayTree):
    x: jnp.ndarray = leaf(shape=3)
    y: jnp.ndarray = leaf(shape=2)


class World(ArrayTree):
    vel: Vec = node(shape=4)


@pytest.fixture
def ones():
    return Vec.blueprint(shape=2).ones()


@pytest.fixture
def twos(ones):
    return ones * 2.0


# ----------------------------------------------------------------------
# Unary
# ----------------------------------------------------------------------


def test_neg(ones):
    r = -ones
    assert jnp.all(r.x == -1.0)
    assert jnp.all(r.y == -1.0)


def test_pos(ones):
    r = +ones
    assert jnp.all(r.x == 1.0)


def test_abs(ones):
    r = abs(-ones)
    assert jnp.all(r.x == 1.0)


# ----------------------------------------------------------------------
# Binary arithmetic — scalar rhs
# ----------------------------------------------------------------------


def test_add_scalar(ones):
    assert jnp.all((ones + 1.0).x == 2.0)


def test_radd_scalar(ones):
    assert jnp.all((1.0 + ones).x == 2.0)


def test_sub_scalar(ones):
    assert jnp.all((ones - 1.0).x == 0.0)


def test_rsub_scalar(ones):
    assert jnp.all((3.0 - ones).x == 2.0)


def test_mul_scalar(ones):
    assert jnp.all((ones * 3.0).x == 3.0)


def test_rmul_scalar(ones):
    assert jnp.all((3.0 * ones).x == 3.0)


def test_truediv_scalar(ones):
    assert jnp.allclose((ones / 2.0).x, 0.5)


def test_rtruediv_scalar(ones):
    assert jnp.allclose((2.0 / ones).x, 2.0)


def test_floordiv_scalar(ones):
    assert jnp.all((ones * 7 // 3).x == 2.0)


def test_rfloordiv_scalar(ones):
    assert jnp.all((7.0 // (ones * 3)).x == 2.0)


def test_mod_scalar(ones):
    assert jnp.all((ones * 7 % 3).x == 1.0)


def test_rmod_scalar(ones):
    assert jnp.all((7.0 % (ones * 3)).x == 1.0)


def test_pow_scalar(ones):
    r = (ones * 3) ** 2
    assert jnp.all(r.x == 9.0)


def test_rpow_scalar(ones):
    assert jnp.allclose((2.0**ones).x, 2.0)


# ----------------------------------------------------------------------
# Binary arithmetic — tree rhs
# ----------------------------------------------------------------------


def test_add_tree(ones, twos):
    assert jnp.all((ones + twos).x == 3.0)


def test_sub_tree(ones, twos):
    assert jnp.all((twos - ones).x == 1.0)


def test_mul_tree(ones, twos):
    assert jnp.all((ones * twos).x == 2.0)


def test_truediv_tree(ones, twos):
    assert jnp.allclose((twos / ones).x, 2.0)


# ----------------------------------------------------------------------
# Comparison
# ----------------------------------------------------------------------


def test_lt(ones):
    r = ones < 2.0
    assert jnp.all(r.x) and jnp.all(r.y)


def test_le(ones):
    r = ones <= 1.0
    assert jnp.all(r.x) and jnp.all(r.y)


def test_gt(ones):
    r = ones > 0.0
    assert jnp.all(r.x) and jnp.all(r.y)


def test_ge(ones):
    r = ones >= 1.0
    assert jnp.all(r.x) and jnp.all(r.y)


# ----------------------------------------------------------------------
# Nested tree
# ----------------------------------------------------------------------


def test_nested_ops():
    w = World.blueprint(shape=2).ones()
    r = (w * 3.0 + 1.0) * 2.0
    assert jnp.all(r.vel.x == 8.0)

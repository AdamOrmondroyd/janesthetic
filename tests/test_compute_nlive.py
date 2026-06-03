import jax
from jax import numpy as jnp
from janesthetic.janesthetic import compute_nlive


def test_compute_nlive_constant():
    """Simple case: all points born from prior (NaN birth), dying in order."""
    logl = jnp.arange(10, dtype=float)
    logl_birth = jnp.full(10, jnp.nan)
    nlive = compute_nlive(logl, logl_birth)
    assert jnp.all(nlive == jnp.arange(10, 0, -1))


def test_compute_nlive_single_replacement():
    """One point dies and is replaced, then all die without replacement."""
    # 3 initial live points with logl = [1, 2, 3], born from prior
    # Point at logl=1 dies, replaced by point born at logl=1 with logl=4
    # Then final 3 live points [2, 3, 4] die without replacement
    logl = jnp.array([1.0, 2.0, 3.0, 4.0])
    logl_birth = jnp.array([jnp.nan, jnp.nan, jnp.nan, 1.0])
    nlive = compute_nlive(logl, logl_birth)
    expected = jnp.array([3, 3, 2, 1])
    assert jnp.all(nlive == expected)


def test_compute_nlive_batch_deletion():
    """Batch of 2 deleted from 4 live points, then final 4 die."""
    # 4 initial points: logl = [1, 2, 3, 4]
    # Kill 2 lowest (logl=1, 2), replace with 2 born at logl=2
    # New points have logl = [5, 6]
    # Then all 4 remaining [3, 4, 5, 6] die without replacement
    logl = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    logl_birth = jnp.array([jnp.nan, jnp.nan, jnp.nan, jnp.nan, 2.0, 2.0])
    nlive = compute_nlive(logl, logl_birth)
    expected = jnp.array([4, 3, 4, 3, 2, 1])
    assert jnp.all(nlive == expected)


def test_compute_nlive_unsorted_input():
    """compute_nlive should work on unsorted logl (returns in input order)."""
    logl = jnp.array([3.0, 1.0, 2.0])
    logl_birth = jnp.full(3, jnp.nan)
    nlive = compute_nlive(logl, logl_birth)
    # Point at logl=3: 1 alive, logl=1: 3 alive, logl=2: 2 alive
    expected = jnp.array([1, 3, 2])
    assert jnp.all(nlive == expected)


def test_compute_nlive_vmappable():
    """compute_nlive should work under vmap."""
    logl = jnp.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]])
    logl_birth = jnp.full((2, 3), jnp.nan)
    nlive = jax.vmap(compute_nlive)(logl, logl_birth)
    expected = jnp.array([[3, 2, 1],
                          [3, 2, 1]])
    assert jnp.all(nlive == expected)

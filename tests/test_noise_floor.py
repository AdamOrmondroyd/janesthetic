"""Cross-implementation agreement on a noisier chain.

Uses anesthetic.examples.perfect_ns.gaussian at lower nlive and higher
dimensionality than the fixture in test_janesthetic.py, so the autodiff path
for d_G has a chance to drift from anesthetic's direct variance formula.
"""
import jax.numpy as jnp
import pytest
from anesthetic.examples.perfect_ns import gaussian as perfect_gaussian

from janesthetic import sort


@pytest.fixture(scope="module")
def noisy_chain():
    return perfect_gaussian(nlive=50, ndims=5)


@pytest.fixture(scope="module")
def noisy_samples(noisy_chain, as_ns_run):
    return sort(as_ns_run(noisy_chain))


@pytest.mark.parametrize("stat,tol", [
    ("logZ", 1e-5),
    ("logL_P", 1e-5),
    ("D_KL", 1e-5),
    ("d_G", 1e-5),
])
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_stat_matches_anesthetic_noisy(noisy_chain, noisy_samples,
                                       stat, tol, beta):
    jan_val = getattr(noisy_samples, stat)(beta)
    ans_val = getattr(noisy_chain, stat)(beta=beta)
    assert jnp.allclose(jan_val, ans_val, atol=tol)

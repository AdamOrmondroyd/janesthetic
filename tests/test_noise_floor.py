"""Cross-implementation agreement on a noisier chain.

Uses anesthetic.examples.perfect_ns.gaussian at lower nlive and higher
dimensionality than the fixture in test_janesthetic.py, so the autodiff path
for d_G has a chance to drift from anesthetic's direct variance formula.
"""
import jax
import jax.numpy as jnp
import pytest
from anesthetic.examples.perfect_ns import gaussian as perfect_gaussian

from janesthetic import sort


@pytest.fixture(scope="module")
def noisy_chain():
    return perfect_gaussian(nlive=50, ndims=5)


@pytest.fixture(scope="module")
def noisy_samples(noisy_chain, as_ns_run):
    with jax.enable_x64():
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
    """Does janesthetic's formula equal anesthetic's?

    Run in x64 because that is a question about the mathematics, not about
    float32: in float32 the cumsum in logdX accumulates up to ~4e-4 of error
    whose size depends on XLA's reduction order, so arm64 and x86 disagree by
    far more than any real drift would. float32 is exercised by the jit/vmap
    tests in test_janesthetic.py.
    """
    with jax.enable_x64():
        jan_val = getattr(noisy_samples, stat)(beta)
        ans_val = getattr(noisy_chain, stat)(beta=beta)
        assert jnp.allclose(jan_val, ans_val, atol=tol)

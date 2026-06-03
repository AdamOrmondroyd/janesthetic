import anesthetic
import jax
import numpy as np
from jax import numpy as jnp
from jax.random import PRNGKey, uniform
import pytest

from janesthetic import D_KL, logL_P, logw, logZ, sort, SortedRun
from janesthetic.special import logsumexp
from midas.nested_sampling import nested_sampling

SIGMA = 0.1
HALF_WIDTH = 10.0
BOX = 2 * HALF_WIDTH


def _gaussian_in_box_ns(nlive=500, seed=42):
    """1D Gaussian likelihood at 0 with uniform box prior on [-10, 10]."""
    def sample_prior(key, n):
        return uniform(key, (n,)) * BOX - HALF_WIDTH

    def log_prior(theta):
        return jnp.where((-HALF_WIDTH < theta) & (theta < HALF_WIDTH),
                         0.0, -jnp.inf)

    def loglikelihood(theta):
        return -theta**2 / (2*SIGMA**2) - 0.5*jnp.log(2*jnp.pi*SIGMA**2)

    return nested_sampling(
        PRNGKey(seed),
        nlive=nlive,
        sample_prior=sample_prior,
        log_prior=log_prior,
        loglikelihood=loglikelihood,
        desc="Gaussian-in-box NS",
    )


def _analytic(beta):
    """Closed-form logZ, logL_P, D_KL for the Gaussian-in-box
    (β > 0, Gaussian contained).

    logL_P uses anesthetic's convention: logL_P = β·<logL>_{P_β}, so that
    D_KL = logL_P - logZ holds uniformly across β.
    """
    log_2pi_sigma2 = jnp.log(2*jnp.pi*SIGMA**2)
    logZ = (1 - beta) / 2 * log_2pi_sigma2 - 0.5 * jnp.log(beta) - jnp.log(BOX)
    logL_P = -0.5 * beta * log_2pi_sigma2 - 0.5
    D_KL = logL_P - logZ
    return logZ, logL_P, D_KL


@pytest.fixture(scope="module")
def ns_run():
    return _gaussian_in_box_ns()


@pytest.fixture(scope="module")
def samples(ns_run):
    return sort(ns_run)


@pytest.fixture(scope="module")
def anesthetic_samples(ns_run):
    data = np.asarray(ns_run.particles.position)[:, None]
    return anesthetic.NestedSamples(
        data=data,
        logL=np.asarray(ns_run.particles.loglikelihood),
        logL_birth=np.asarray(ns_run.particles.loglikelihood_birth),
    )


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_match_analytic(samples, beta):
    analytic_logZ, analytic_logL_P, analytic_D_KL = _analytic(beta)
    assert jnp.allclose(samples.logZ(beta), analytic_logZ, atol=0.3)
    assert jnp.allclose(samples.logL_P(beta), analytic_logL_P, atol=0.3)
    assert jnp.allclose(samples.D_KL(beta), analytic_D_KL, atol=0.3)


def test_D_KL_zero_beta(samples):
    """β=0: posterior collapses to prior, so D_KL = 0."""
    assert jnp.abs(samples.D_KL(0.0)) < 1e-3


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_logZ_matches_anesthetic(samples, anesthetic_samples, beta):
    """Same ns_run → same logZ to float-precision tolerance."""
    assert jnp.allclose(samples.logZ(beta),
                        anesthetic_samples.logZ(beta=beta),
                        atol=1e-5)


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_D_KL_matches_anesthetic(samples, anesthetic_samples, beta):
    """Same ns_run => same D_KL to float-precision tolerance."""
    assert jnp.allclose(samples.D_KL(beta),
                        anesthetic_samples.D_KL(beta=beta),
                        atol=1e-5)


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_logL_P_matches_anesthetic(samples, anesthetic_samples, beta):
    """Same ns_run => same logL_P (β·<logL>_Pβ convention)."""
    assert jnp.allclose(samples.logL_P(beta),
                        anesthetic_samples.logL_P(beta=beta),
                        atol=1e-5)


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_free_functions_match_methods(samples, beta):
    assert jnp.allclose(logw(samples, beta), samples.logw(beta))
    assert jnp.allclose(logZ(samples, beta), samples.logZ(beta))
    assert jnp.allclose(logL_P(samples, beta), samples.logL_P(beta))
    assert jnp.allclose(D_KL(samples, beta), samples.D_KL(beta))


def test_vmap_over_sortedrun(samples):
    stacked = jax.tree.map(lambda x: jnp.stack([x, x]), samples)
    out_logZ = jax.vmap(logZ)(stacked)
    out_D_KL = jax.vmap(D_KL)(stacked)
    assert out_logZ.shape == (2,)
    assert jnp.allclose(out_logZ, samples.logZ())
    assert jnp.allclose(out_D_KL, samples.D_KL())


def test_logZ_grad_safe_with_neginf_logl():
    """logw keeps value_and_grad(logZ) finite when logl -inf."""
    nlive = jnp.array([5, 4, 3, 2, 1])
    logl = jnp.array([-jnp.inf, 0.0, 1.0, 2.0, 3.0])

    def logZ_fn(beta):
        return logsumexp(SortedRun(nlive=nlive, logl=logl).logw(beta))

    for beta in [0.0, 0.5, 1.0, 2.0]:
        Z, dZ = jax.value_and_grad(logZ_fn)(beta)
        assert jnp.isfinite(Z), f"logZ not finite at β={beta}"
        assert jnp.isfinite(dZ), f"d logZ/dβ not finite at β={beta}"

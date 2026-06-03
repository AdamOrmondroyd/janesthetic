import anesthetic
import jax
import numpy as np
from jax import numpy as jnp
from jax.random import PRNGKey, uniform
import pytest

from janesthetic import D_KL, d_G, logL_P, logw, logZ, sort, SortedRun
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
    """Closed-form logZ, logL_P, D_KL, d_G for the Gaussian-in-box
    (β > 0, Gaussian contained).

    logL_P uses anesthetic's convention: logL_P = β·<logL>_{P_β}, so that
    D_KL = logL_P - logZ holds uniformly across β.

    d_G = 2β²·Var_{P_β}(logL). For 1D Gaussian-in-box, Var_{P_β}(logL) =
    1/(2β²) so d_G = 1 independent of β.
    """
    log_2pi_sigma2 = jnp.log(2*jnp.pi*SIGMA**2)
    logZ = (1 - beta) / 2 * log_2pi_sigma2 - 0.5 * jnp.log(beta) - jnp.log(BOX)
    logL_P = -0.5 * beta * log_2pi_sigma2 - 0.5
    D_KL = logL_P - logZ
    d_G = 1.0
    return logZ, logL_P, D_KL, d_G


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
    analytic_logZ, analytic_logL_P, analytic_D_KL, analytic_d_G = _analytic(beta)
    assert jnp.allclose(samples.logZ(beta), analytic_logZ, atol=0.3)
    assert jnp.allclose(samples.logL_P(beta), analytic_logL_P, atol=0.3)
    assert jnp.allclose(samples.D_KL(beta), analytic_D_KL, atol=0.3)
    assert jnp.allclose(samples.d_G(beta), analytic_d_G, atol=0.3)


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
def test_d_G_matches_anesthetic(samples, anesthetic_samples, beta):
    """Same ns_run => same d_G (Bayesian model dimensionality)."""
    assert jnp.allclose(samples.d_G(beta),
                        anesthetic_samples.d_G(beta=beta),
                        atol=1e-5)


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_free_functions_match_methods(samples, beta):
    assert jnp.allclose(logw(samples, beta), samples.logw(beta))
    assert jnp.allclose(logZ(samples, beta), samples.logZ(beta))
    assert jnp.allclose(logL_P(samples, beta), samples.logL_P(beta))
    assert jnp.allclose(D_KL(samples, beta), samples.D_KL(beta))
    assert jnp.allclose(d_G(samples, beta), samples.d_G(beta))


def test_vmap_over_sortedrun(samples):
    """vmap of all four free functions over a non-degenerate 2-batch."""
    other = SortedRun(logl=samples.logl + 0.5, nlive=samples.nlive)
    stacked = jax.tree.map(lambda x, y: jnp.stack([x, y]), samples, other)
    n = samples.logl.shape[0]

    out_logw = jax.vmap(logw)(stacked)
    out_logZ = jax.vmap(logZ)(stacked)
    out_logL_P = jax.vmap(logL_P)(stacked)
    out_D_KL = jax.vmap(D_KL)(stacked)
    out_d_G = jax.vmap(d_G)(stacked)

    assert out_logw.shape == (2, n)
    assert out_logZ.shape == (2,)
    assert out_logL_P.shape == (2,)
    assert out_D_KL.shape == (2,)
    assert out_d_G.shape == (2,)

    assert jnp.allclose(out_logw[0], samples.logw())
    assert jnp.allclose(out_logw[1], other.logw())
    assert jnp.allclose(out_logZ[0], samples.logZ())
    assert jnp.allclose(out_logZ[1], other.logZ())
    assert jnp.allclose(out_logL_P[0], samples.logL_P())
    assert jnp.allclose(out_logL_P[1], other.logL_P())
    assert jnp.allclose(out_D_KL[0], samples.D_KL())
    assert jnp.allclose(out_D_KL[1], other.D_KL())
    assert jnp.allclose(out_d_G[0], samples.d_G())
    assert jnp.allclose(out_d_G[1], other.d_G())

    # Sanity: the two batch elements are genuinely different.
    assert not jnp.allclose(out_logZ[0], out_logZ[1])


def test_vmap_over_beta(samples):
    """vmap of the methods over a beta schedule."""
    betas = jnp.array([0.5, 1.0, 2.0])
    n = samples.logl.shape[0]

    out_logw = jax.vmap(samples.logw)(betas)
    out_logZ = jax.vmap(samples.logZ)(betas)
    out_logL_P = jax.vmap(samples.logL_P)(betas)
    out_D_KL = jax.vmap(samples.D_KL)(betas)
    out_d_G = jax.vmap(samples.d_G)(betas)

    assert out_logw.shape == (3, n)
    assert out_logZ.shape == (3,)
    assert out_logL_P.shape == (3,)
    assert out_D_KL.shape == (3,)
    assert out_d_G.shape == (3,)

    for i, beta in enumerate([0.5, 1.0, 2.0]):
        assert jnp.allclose(out_logw[i], samples.logw(beta))
        assert jnp.allclose(out_logZ[i], samples.logZ(beta))
        assert jnp.allclose(out_logL_P[i], samples.logL_P(beta))
        assert jnp.allclose(out_D_KL[i], samples.D_KL(beta))
        assert jnp.allclose(out_d_G[i], samples.d_G(beta))


def test_double_vmap_samples_and_beta(samples):
    """vmap over both SortedRun and beta axes simultaneously."""
    other = SortedRun(logl=samples.logl + 0.5, nlive=samples.nlive)
    stacked = jax.tree.map(lambda x, y: jnp.stack([x, y]), samples, other)
    betas = jnp.array([0.5, 1.0, 2.0])

    out = jax.vmap(jax.vmap(logZ, in_axes=(None, 0)),
                   in_axes=(0, None))(stacked, betas)
    assert out.shape == (2, 3)
    for i, run in enumerate([samples, other]):
        for j, beta in enumerate([0.5, 1.0, 2.0]):
            assert jnp.allclose(out[i, j], run.logZ(beta))


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_jit_free_functions(samples, beta):
    """jit each free function, compare to the un-jitted method."""
    assert jnp.allclose(jax.jit(logw)(samples, beta), samples.logw(beta))
    assert jnp.allclose(jax.jit(logZ)(samples, beta), samples.logZ(beta))
    assert jnp.allclose(jax.jit(logL_P)(samples, beta), samples.logL_P(beta))
    assert jnp.allclose(jax.jit(D_KL)(samples, beta), samples.D_KL(beta))
    assert jnp.allclose(jax.jit(d_G)(samples, beta), samples.d_G(beta))


def test_jit_vmap_composition(samples):
    """jit(vmap(...)) — the actual production pattern."""
    other = SortedRun(logl=samples.logl + 0.5, nlive=samples.nlive)
    stacked = jax.tree.map(lambda x, y: jnp.stack([x, y]), samples, other)

    out_logZ = jax.jit(jax.vmap(logZ))(stacked)
    out_logL_P = jax.jit(jax.vmap(logL_P))(stacked)
    out_D_KL = jax.jit(jax.vmap(D_KL))(stacked)
    out_d_G = jax.jit(jax.vmap(d_G))(stacked)

    assert out_logZ.shape == (2,)
    assert jnp.allclose(out_logZ[0], samples.logZ())
    assert jnp.allclose(out_logZ[1], other.logZ())
    assert jnp.allclose(out_logL_P[0], samples.logL_P())
    assert jnp.allclose(out_logL_P[1], other.logL_P())
    assert jnp.allclose(out_D_KL[0], samples.D_KL())
    assert jnp.allclose(out_D_KL[1], other.D_KL())
    assert jnp.allclose(out_d_G[0], samples.d_G())
    assert jnp.allclose(out_d_G[1], other.d_G())


def test_logZ_grad_safe_with_neginf_logl():
    """logw keeps logZ and its first two β-derivatives finite when logl -inf.

    The second-derivative assertion guards d_G (which is 2β²·d²logZ/dβ²) against
    NaN propagation through the masked -inf branch in `logw`.
    """
    nlive = jnp.array([5, 4, 3, 2, 1])
    logl = jnp.array([-jnp.inf, 0.0, 1.0, 2.0, 3.0])

    def logZ_fn(beta):
        return logsumexp(SortedRun(nlive=nlive, logl=logl).logw(beta))

    for beta in [0.0, 0.5, 1.0, 2.0]:
        Z, dZ = jax.value_and_grad(logZ_fn)(beta)
        ddZ = jax.grad(jax.grad(logZ_fn))(beta)
        assert jnp.isfinite(Z), f"logZ not finite at β={beta}"
        assert jnp.isfinite(dZ), f"d logZ/dβ not finite at β={beta}"
        assert jnp.isfinite(ddZ), f"d² logZ/dβ² not finite at β={beta}"

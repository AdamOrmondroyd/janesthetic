"""A `keep` mask must be exactly equivalent to deleting the masked rows.

Zeroing both the birth and the death event of a row removes that particle from
the run outright, so `sort(run, keep)` is not an approximation to
`sort(filtered_run)` -- it is the same quadrature on the same points, and the
tolerances here are float noise rather than statistical slack.

Run in x64 for the same reason as test_noise_floor.py: this is a question about
the mathematics, not about float32.
"""
import jax
import numpy as np
import pytest
from anesthetic.examples.perfect_ns import gaussian as perfect_gaussian
from blackjax.ns.base import NSState, StateWithLogLikelihood
from jax import numpy as jnp

from janesthetic import D_KL, SortedRun, d_G, logL_P, logZ, sort

BETAS = [0.5, 1.0, 2.0]
STATS = ["logZ", "logL_P", "D_KL", "d_G"]
ATOL = 1e-10


def _ns_run(logl, logl_birth):
    """The minimal state `sort` reads -- see the `as_ns_run` fixture."""
    return NSState(particles=StateWithLogLikelihood(
        position=jnp.zeros((logl.shape[0], 1)),
        logdensity=jnp.full_like(logl, jnp.nan),
        loglikelihood=logl,
        loglikelihood_birth=logl_birth,
    ))


@pytest.fixture(scope="module")
def chain():
    return perfect_gaussian(nlive=200, ndims=2, sigma=0.1, R=1.0)


@pytest.fixture(scope="module")
def arrays(chain):
    return jnp.asarray(chain.logL.values), jnp.asarray(chain.logL_birth.values)


def _masked_and_filtered(arrays, frac, pad=None):
    """`sort` with `frac` of the rows masked, and with them dropped instead."""
    logl, logl_birth = arrays
    keep = jnp.asarray(np.random.default_rng().random(logl.shape[0]) >= frac)
    if pad is not None:
        logl = jnp.where(keep, logl, pad)
        logl_birth = jnp.where(keep, logl_birth, pad)
    masked = sort(_ns_run(logl, logl_birth), keep)
    order = jnp.argsort(logl)
    filtered = sort(_ns_run(logl[order][keep[order]],
                            logl_birth[order][keep[order]]))
    return masked, filtered


@pytest.mark.parametrize("frac", [0.0, 0.1, 0.35])
def test_masked_run_matches_filtered(arrays, frac):
    """Per-point quantities agree wherever the mask keeps a row."""
    with jax.enable_x64():
        masked, filtered = _masked_and_filtered(arrays, frac)
        assert jnp.all(masked.nlive[masked.keep] == filtered.nlive)
        assert jnp.all(masked.logl[masked.keep] == filtered.logl)
        assert jnp.allclose(masked.logdX()[masked.keep], filtered.logdX(),
                            atol=ATOL)


@pytest.mark.parametrize("frac", [0.0, 0.1, 0.35])
@pytest.mark.parametrize("beta", BETAS)
@pytest.mark.parametrize("stat", STATS)
def test_masked_stats_match_filtered(arrays, frac, beta, stat):
    with jax.enable_x64():
        masked, filtered = _masked_and_filtered(arrays, frac)
        assert jnp.allclose(getattr(masked, stat)(beta),
                            getattr(filtered, stat)(beta), atol=ATOL)


@pytest.mark.parametrize("pad", [jnp.nan, -jnp.inf])
@pytest.mark.parametrize("stat", STATS)
def test_padded_rows_carry_nonsense(arrays, pad, stat):
    """The masked rows may hold anything -- nan is what midas would send.

    Nonsense logl also moves the argsort in `sort`, so this checks the mask
    survives being permuted differently from the filtered run.
    """
    with jax.enable_x64():
        masked, filtered = _masked_and_filtered(arrays, 0.1, pad=pad)
        assert jnp.isfinite(getattr(masked, stat)())
        assert jnp.allclose(getattr(masked, stat)(), getattr(filtered, stat)(),
                            atol=ATOL)


@pytest.mark.parametrize("stat", STATS)
def test_all_true_mask_matches_no_mask(arrays, stat):
    """A mask that keeps everything reduces to the unmasked path."""
    with jax.enable_x64():
        unmasked = sort(_ns_run(*arrays))
        masked = sort(_ns_run(*arrays), jnp.ones(arrays[0].shape[0], bool))
        assert jnp.all(masked.nlive == unmasked.nlive)
        assert jnp.allclose(getattr(masked, stat)(), getattr(unmasked, stat)(),
                            atol=ATOL)


@pytest.fixture(scope="module")
def stacked(arrays):
    """Two runs with *different* masks, stacked along a batch axis."""
    logl, logl_birth = arrays
    rng = np.random.default_rng()
    runs = [sort(_ns_run(logl, logl_birth),
                 jnp.asarray(rng.random(logl.shape[0]) >= frac))
            for frac in (0.1, 0.35)]
    return runs, jax.tree.map(lambda x, y: jnp.stack([x, y]), *runs)


@pytest.mark.parametrize("fn", [logZ, logL_P, D_KL, d_G])
def test_vmap_over_masked_runs(stacked, fn):
    runs, batch = stacked
    out = jax.vmap(fn)(batch)
    assert out.shape == (2,)
    for i, run in enumerate(runs):
        assert jnp.allclose(out[i], fn(run))


@pytest.mark.parametrize("fn", [logZ, logL_P, D_KL, d_G])
def test_jit_vmap_over_masked_runs(stacked, fn):
    runs, batch = stacked
    out = jax.jit(jax.vmap(fn))(batch)
    for i, run in enumerate(runs):
        assert jnp.allclose(out[i], fn(run), rtol=1e-4)


def test_grad_safe_with_masked_nans():
    """logZ and its first two beta-derivatives survive nan in the masked rows.

    d_G is 2*beta**2 * d2logZ/dbeta2, so a nan leaking out of the masked branch
    would surface there rather than in logZ.
    """
    keep = jnp.array([True, False, True, True, False, True])
    logl = jnp.where(keep, jnp.array([0.0, 0.0, 1.0, 2.0, 0.0, 3.0]), jnp.nan)
    run = SortedRun(logl=logl, nlive=jnp.array([6, 5, 4, 3, 2, 1]), keep=keep)
    for beta in [0.0, 0.5, 1.0, 2.0]:
        assert jnp.isfinite(run.logZ(beta)), f"logZ not finite at beta={beta}"
        assert jnp.isfinite(run.logL_P(beta)), f"logL_P not finite at {beta}"
        assert jnp.isfinite(run.d_G(beta)), f"d_G not finite at beta={beta}"

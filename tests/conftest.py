"""Shared fixtures for turning anesthetic runs into janesthetic inputs."""
import jax.numpy as jnp
import pytest
from blackjax.ns.base import NSState, StateWithLogLikelihood


@pytest.fixture(scope="session")
def as_ns_run():
    """Wrap anesthetic NestedSamples in blackjax's nested sampling state.

    `sort` reads only the loglikelihood fields. logdensity is blackjax's
    log-density under the *prior* alone (not prior + likelihood — nested
    sampling needs the two kept apart), and an anesthetic chain doesn't carry
    it, hence nan. For the perfect_ns fixtures the prior is uniform on a ball
    of radius R, so it is really the constant -log V_R; fill that in if a
    test ever needs the field.
    """
    def _wrap(chain):
        params = [c for c in chain.columns if not isinstance(c, str)]
        loglikelihood = jnp.asarray(chain.logL.values)
        return NSState(particles=StateWithLogLikelihood(
            position=jnp.asarray(chain[params].values),
            logdensity=jnp.full_like(loglikelihood, jnp.nan),
            loglikelihood=loglikelihood,
            loglikelihood_birth=jnp.asarray(chain.logL_birth.values),
        ))
    return _wrap

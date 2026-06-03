import numpy as np
from jax import numpy as jnp
from anesthetic.utils import logsumexp as anesthetic_logsumexp

from janesthetic.special import logsumexp


def test_logsumexp_matches_anesthetic():
    np.random.seed(0)
    a = np.random.rand(10)
    b = np.random.rand(10)
    assert logsumexp(-jnp.inf, b=jnp.array([-jnp.inf])) == -jnp.inf
    assert jnp.allclose(logsumexp(a, b=b), anesthetic_logsumexp(a, b=b))
    a[0] = -np.inf
    assert jnp.allclose(logsumexp(a, b=b), anesthetic_logsumexp(a, b=b))
    b[0] = -np.inf
    assert jnp.allclose(logsumexp(a, b=b), anesthetic_logsumexp(a, b=b))

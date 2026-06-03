from jax import numpy as jnp
from jax.scipy import special


def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
    """Compute the log of the sum of exponentials of input elements.

    Blatantly copied from anesthetic.
    """
    if b is None:
        b = jnp.ones_like(a)
    b = jnp.where(a == -jnp.inf, 0, b)
    return special.logsumexp(a, axis=axis, b=b, keepdims=keepdims,
                             return_sign=return_sign)

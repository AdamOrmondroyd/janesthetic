from chex import dataclass
from jax import Array, grad
from jax import numpy as jnp
from janesthetic.special import logsumexp


def compute_nlive(logl, logl_birth):
    n = logl.shape[0]

    # Birth events (+1) at indices 0..n-1, death events (-1) at indices n..2n-1
    combined_logL = jnp.concatenate([logl_birth, logl])
    combined_n = jnp.concatenate([jnp.ones(n, dtype=int),
                                  -jnp.ones(n, dtype=int)])

    # Sort: nans first (initial live points,
    # mirrors anesthetic's na_position='first'),
    # then by logL ascending, then deaths (-1) before births (+1) at ties.
    is_nan = jnp.isnan(combined_logL)
    safe_logL = jnp.where(is_nan, -jnp.inf, combined_logL)
    sorted_indices = jnp.lexsort((combined_n, safe_logL, ~is_nan))
    sorted_n = combined_n[sorted_indices]

    cumsum = jnp.maximum(jnp.cumsum(sorted_n), 0)

    # Inverse permutation: maps combined index -> sorted position
    inv_perm = jnp.argsort(sorted_indices)

    # Death events are combined indices n..2n-1
    num_live = cumsum[inv_perm[n:]] + 1

    return num_live


def sort(ns_run):
    sort_idx = jnp.argsort(ns_run.particles.loglikelihood)
    logl = ns_run.particles.loglikelihood[sort_idx]
    logl_birth = ns_run.particles.loglikelihood_birth[sort_idx]
    nlive = compute_nlive(logl, logl_birth)
    return SortedRun(logl=logl, logl_birth=logl_birth, nlive=nlive)


@dataclass
class SortedRun:
    logl: Array
    logl_birth: Array
    nlive: Array

    def logw(self, beta=1.0):
        """
        Log nested sampling weights. Blatantly stolen from anesthetic.
        """
        t = jnp.log(self.nlive/(self.nlive+1))
        logX = jnp.cumsum(t)
        logXp = jnp.concatenate([jnp.array([0.0]), logX[:-1]])
        logXm = jnp.concatenate([logX[1:], jnp.array([-jnp.inf])])
        logdX = jnp.log1p(-jnp.exp(logXm-logXp)) + logXp - jnp.log(2)
        safe_logl = jnp.where(jnp.isneginf(self.logl), 0.0, self.logl)
        return jnp.where(
            jnp.isneginf(self.logl),
            -jnp.inf,
            logdX + safe_logl * beta
        )

    def logZ(self, beta=1.0):
        return logsumexp(self.logw(beta))

    def logL_P(self, beta=1.0):
        return beta * grad(self.logZ)(beta)

    def D_KL(self, beta=1.0):
        return self.logL_P(beta) - self.logZ(beta)


def logw(sorted_run: SortedRun, beta=1.0): return sorted_run.logw(beta)
def logZ(sorted_run: SortedRun, beta=1.0): return sorted_run.logZ(beta)
def logL_P(sorted_run: SortedRun, beta=1.0): return sorted_run.logL_P(beta)
def D_KL(sorted_run: SortedRun, beta=1.0): return sorted_run.D_KL(beta)

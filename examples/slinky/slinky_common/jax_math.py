"""JAX-only numerical helpers duplicated across many stiffness-parameterization notebooks."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def inv_softplus(y: jax.Array) -> jax.Array:
    """
    Inverse of the softplus function: find ``x`` such that ``softplus(x) == y``.

    For ``y > 0``, ``inv_softplus(y) = log(exp(y) - 1)``, implemented as ``log(expm1(y))``
    for numerical stability when ``y`` is small.

    Used when storing positive stiffness-related parameters in unconstrained space:
    ``positive_value = softplus(unconstrained)``.
    """
    return jnp.log(jnp.expm1(y))

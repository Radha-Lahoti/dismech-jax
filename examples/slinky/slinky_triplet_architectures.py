"""
Learned triplet stiffness models used by the 3D slinky / ribbon notebooks.

Each class subclasses :class:`util.TripletModel`.  Pick one architecture in a thin notebook::

    from slinky_triplet_architectures import Triplet3DCholeskyMLP
    from util import validate_model, train_model

    validate_model(Triplet3DCholeskyMLP, der_K=init_K)
    train_model(Triplet3DCholeskyMLP, init_K=init_K, alpha_spec=0.01)

Available classes include ``Triplet2DStiffnessMLP``, ``Triplet2DKPairSoftplusMLP`` (scalar
``K(eps)`` single-trajectory style), ``Triplet2DQuadraticPlusMLP``, ``TripletSlinkyPureMLP``,
``Triplet3DCholeskyConstant``, ``Triplet3DCholeskyMLP``, and ``Triplet3DBoundedSpectral``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float

from slinky_common.jax_math import inv_softplus
from util import TripletModel


# ---------------------------------------------------------------------------
# Shared MLP heads
# ---------------------------------------------------------------------------


class MLPScalarSoftplus(eqx.Module):
    """Two-layer MLP producing one nonnegative scalar (softplus output)."""

    layer1: eqx.nn.Linear
    layer2: eqx.nn.Linear

    def __init__(self, in_features: int, hidden_size: int, key: jax.Array):
        key1, key2 = jax.random.split(key)
        self.layer1 = eqx.nn.Linear(in_features, hidden_size, key=key1)
        self.layer2 = eqx.nn.Linear(hidden_size, 1, key=key2)
        self.layer1 = eqx.tree_at(lambda l: l.weight, self.layer1, self.layer1.weight * 1e-2)
        self.layer2 = eqx.tree_at(lambda l: l.weight, self.layer2, self.layer2.weight * 1e-2)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jax.nn.softplus(self.layer1(x))
        x = self.layer2(x)
        return jax.nn.softplus(x[0])


class MLPTanhVector(eqx.Module):
    """Two-layer MLP with tanh activation; vector-valued output (e.g. 3 corrections)."""

    layer1: eqx.nn.Linear
    layer2: eqx.nn.Linear

    def __init__(self, in_features: int, hidden_size: int, out_features: int, key: jax.Array):
        key1, key2 = jax.random.split(key)
        self.layer1 = eqx.nn.Linear(in_features, hidden_size, key=key1)
        self.layer2 = eqx.nn.Linear(hidden_size, out_features, key=key2)
        self.layer1 = eqx.tree_at(lambda l: l.weight, self.layer1, self.layer1.weight * 1e-2)
        self.layer2 = eqx.tree_at(lambda l: l.weight, self.layer2, self.layer2.weight * 1e-2)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.ravel(x)
        x = jax.nn.tanh(self.layer1(x))
        return self.layer2(x)


# ---------------------------------------------------------------------------
# 2D stiffness + small MLP correction (most ``K_is_2_dim`` notebooks)
# ---------------------------------------------------------------------------


class Triplet2DStiffnessMLP(TripletModel):
    """Baseline stretching + bending with a nonnegative scalar MLP correction."""

    K: jax.Array
    mlp: MLPScalarSoftplus

    def __init__(self, der_K: jax.Array, key: jax.Array):
        self.K = der_K
        self.mlp = MLPScalarSoftplus(5, 10, key)

    def __call__(self, del_strain: Float[jax.Array, "5"]) -> Float[jax.Array, ""]:
        stretching_energy = 0.5 * (
            self.K[0] * del_strain[0] ** 2 + self.K[0] * del_strain[1] ** 2
        )
        bending_energy = 0.5 * (self.K[1] * del_strain[3] ** 2)
        return stretching_energy + bending_energy + self.mlp(del_strain)


class Triplet2DKPairSoftplusMLP(TripletModel):
    """Learnable (k_stretch, k_bend) with strain-dependent softplus correction (``K(eps)`` single-traj notebooks)."""

    K0: jax.Array
    mlp: MLPTanhVector
    corr_scale: float = eqx.field(static=True)

    def __init__(self, der_K: jax.Array, key: jax.Array, corr_scale: float = 1.0):
        der_K = jnp.ravel(der_K)
        if der_K.shape != (2,):
            raise ValueError(f"Expected der_K shape (2,), got {der_K.shape}")
        if jnp.any(der_K <= 0.0):
            raise ValueError("Initial der_K entries must be positive.")
        eps = 1e-6
        self.K0 = jnp.log(jnp.expm1(der_K - eps))
        self.mlp = MLPTanhVector(5, 10, 2, key=key)
        self.corr_scale = float(corr_scale)

    def get_K(self, del_strain: Float[jax.Array, "5"]) -> Float[jax.Array, "2"]:
        del_strain = jnp.ravel(del_strain)
        dK = self.corr_scale * self.mlp(del_strain)
        return jax.nn.softplus(self.K0 + dK) + 1e-6

    def __call__(self, del_strain: Float[jax.Array, "5"]) -> Float[jax.Array, ""]:
        del_strain = jnp.ravel(del_strain)
        K = self.get_K(del_strain)
        stretching_energy = 0.5 * (K[0] * del_strain[0] ** 2 + K[0] * del_strain[1] ** 2)
        bending_energy = 0.5 * K[1] * del_strain[3] ** 2
        return stretching_energy + bending_energy


# ---------------------------------------------------------------------------
# Quadratic baseline + MLP on full strain (older ``3d_slinky.ipynb`` cell)
# ---------------------------------------------------------------------------


class Triplet2DQuadraticPlusMLP(TripletModel):
    """0.5 * sum(K * strain^2) plus scalar MLP on the full 5-vector."""

    K: jax.Array
    mlp: MLPScalarSoftplus

    def __init__(self, der_K: jax.Array, key: jax.Array):
        self.K = der_K
        self.mlp = MLPScalarSoftplus(int(der_K.shape[0]), 10, key)

    def __call__(self, del_strain: Float[jax.Array, "5"]) -> Float[jax.Array, ""]:
        return 0.5 * jnp.sum(self.K * del_strain**2) + self.mlp(del_strain)


# ---------------------------------------------------------------------------
# Pure MLP energy on selected strain components (``10_loop_slinky`` experiment)
# ---------------------------------------------------------------------------


class TripletSlinkyPureMLP(TripletModel):
    """Energy = MLP([e0,e1,eb]); ``K`` stores a dummy stiffness vector for logging."""

    K: jax.Array
    mlp: MLPScalarSoftplus

    def __init__(self, der_K: jax.Array, key: jax.Array):
        self.K = der_K
        self.mlp = MLPScalarSoftplus(3, 20, key)

    def __call__(self, del_strain: Float[jax.Array, "5"]) -> Float[jax.Array, ""]:
        strains_of_use = jnp.concatenate([del_strain[:2], del_strain[3:4]])
        return self.mlp(strains_of_use)


# ---------------------------------------------------------------------------
# 3×3 structured stiffness: Cholesky base only (ribbon ``K_is_3_dim`` without MLP)
# ---------------------------------------------------------------------------


class Triplet3DCholeskyConstant(TripletModel):
    """Constant PSD structured stiffness via 2×2 Cholesky parameters."""

    K: jax.Array

    def __init__(self, der_K: jax.Array, key: jax.Array):
        del key
        der_K = jnp.ravel(der_K)
        if der_K.shape != (3,):
            raise ValueError(f"Expected der_K shape (3,), got {der_K.shape}")

        k_ss0, k_sb0, k_bb0 = der_K
        eps = 1e-6
        if (k_ss0 < 0) or (k_bb0 < 0) or (k_ss0 * k_bb0 - 2.0 * k_sb0**2 < 0):
            raise ValueError(
                "Initial [k_ss, k_sb, k_bb] must satisfy PSD: "
                "k_ss >= 0, k_bb >= 0, k_ss*k_bb - 2*k_sb^2 >= 0."
            )

        l11 = jnp.sqrt(jnp.maximum(k_ss0, eps))
        l21 = jnp.sqrt(2.0) * k_sb0 / l11
        rem = k_bb0 - l21**2
        l22 = jnp.sqrt(jnp.maximum(rem, eps))
        p0 = inv_softplus(l11 - eps)
        p1 = l21
        p2 = inv_softplus(l22 - eps)
        self.K = jnp.array([p0, p1, p2])

    def _vec_to_L(self, p: jax.Array) -> jax.Array:
        eps = 1e-6
        p = jnp.ravel(p)
        return jnp.array(
            [
                [jax.nn.softplus(p[0]) + eps, 0.0],
                [p[1], jax.nn.softplus(p[2]) + eps],
            ]
        )

    def get_K_entries(self, del_strain: Float[jax.Array, "5"] | None = None) -> jax.Array:
        del del_strain
        L = self._vec_to_L(self.K)
        B = L @ L.T
        k_ss = B[0, 0]
        k_sb = B[0, 1] / jnp.sqrt(2.0)
        k_bb = B[1, 1]
        return jnp.array([k_ss, k_sb, k_bb])

    def get_K_matrix(self, del_strain: Float[jax.Array, "5"] | None = None) -> jax.Array:
        k_ss, k_sb, k_bb = self.get_K_entries()
        return jnp.array(
            [
                [k_ss, 0.0, k_sb],
                [0.0, k_ss, k_sb],
                [k_sb, k_sb, k_bb],
            ]
        )

    def __call__(self, del_strain: Float[jax.Array, "5"]) -> Float[jax.Array, ""]:
        del_strain = jnp.ravel(del_strain)
        e0, e1, eb = del_strain[0], del_strain[1], del_strain[3]
        k_ss, k_sb, k_bb = self.get_K_entries()
        return (
            0.5 * k_ss * (e0**2 + e1**2)
            + k_sb * (e0 + e1) * eb
            + 0.5 * k_bb * eb**2
        )


# ---------------------------------------------------------------------------
# Cholesky + MLP correction + l_k scaling (matrix / multiset notebooks)
# ---------------------------------------------------------------------------


class Triplet3DCholeskyMLP(TripletModel):
    """Strain-dependent stiffness with Cholesky PSD guarantee and ``l_k`` scaling."""

    K: jax.Array
    mlp: MLPTanhVector
    l_k: float

    def __init__(self, der_K: jax.Array, key: jax.Array, l_k: float = 0.25):
        der_K = jnp.ravel(der_K)
        if der_K.shape != (3,):
            raise ValueError(f"Expected der_K shape (3,), got {der_K.shape}")

        k_ss0, k_sb0, k_bb0 = der_K
        eps = 1e-6
        if (k_ss0 < 0) or (k_bb0 < 0) or (k_ss0 * k_bb0 - 2.0 * k_sb0**2 < 0):
            raise ValueError(
                "Initial [k_ss, k_sb, k_bb] must satisfy PSD: "
                "k_ss >= 0, k_bb >= 0, k_ss*k_bb - 2*k_sb^2 >= 0."
            )

        l11 = jnp.sqrt(jnp.maximum(k_ss0, eps))
        l21 = jnp.sqrt(2.0) * k_sb0 / l11
        rem = k_bb0 - l21**2
        l22 = jnp.sqrt(jnp.maximum(rem, eps))
        p0 = inv_softplus(l11 - eps)
        p1 = l21
        p2 = inv_softplus(l22 - eps)

        self.K = jnp.array([p0, p1, p2])
        self.mlp = MLPTanhVector(5, 10, 3, key=key)
        self.l_k = float(l_k)

    def _vec_to_L(self, p: jax.Array) -> jax.Array:
        eps = 1e-6
        p = jnp.ravel(p)
        return jnp.array(
            [
                [jax.nn.softplus(p[0]) + eps, 0.0],
                [p[1], jax.nn.softplus(p[2]) + eps],
            ]
        )

    def get_K_entries_base(self, del_strain: Float[jax.Array, "5"]) -> jax.Array:
        del_strain = jnp.ravel(del_strain)
        dp = 1e-2 * self.mlp(del_strain)
        p = self.K + dp
        L = self._vec_to_L(p)
        B = L @ L.T
        k_ss = B[0, 0]
        k_sb = B[0, 1] / jnp.sqrt(2.0)
        k_bb = B[1, 1]
        return jnp.array([k_ss, k_sb, k_bb])

    def get_K_entries(self, del_strain: Float[jax.Array, "5"]) -> jax.Array:
        k_ss_b, k_sb_b, k_bb_b = self.get_K_entries_base(del_strain)
        return jnp.array(
            [self.l_k * k_ss_b, k_sb_b, k_bb_b / self.l_k]
        )

    def get_K_matrix(self, del_strain: Float[jax.Array, "5"]) -> jax.Array:
        k_ss, k_sb, k_bb = self.get_K_entries(del_strain)
        return jnp.array(
            [
                [k_ss, 0.0, k_sb],
                [0.0, k_ss, k_sb],
                [k_sb, k_sb, k_bb],
            ]
        )

    def __call__(self, del_strain: Float[jax.Array, "5"]) -> Float[jax.Array, ""]:
        del_strain = jnp.ravel(del_strain)
        e0, e1, eb = del_strain[0], del_strain[1], del_strain[3]
        k_ss, k_sb, k_bb = self.get_K_entries(del_strain)
        return (
            0.5 * k_ss * (e0**2 + e1**2)
            + k_sb * (e0 + e1) * eb
            + 0.5 * k_bb * eb**2
        )


# ---------------------------------------------------------------------------
# Bounded reparameterization for spectral-spread regularized training
# ---------------------------------------------------------------------------


class Triplet3DBoundedSpectral(TripletModel):
    """Well-conditioned bounded (s, r, β) parameterization + small MLP drift."""

    theta: jax.Array
    mlp: MLPTanhVector
    rho: float = eqx.field(static=True)
    beta_max: float = eqx.field(static=True)
    corr_scale: float = eqx.field(static=True)
    l_k: float = eqx.field(static=True)
    K_log: jax.Array

    def __init__(
        self,
        der_K: jax.Array,
        key: jax.Array,
        l_k: float = 0.1,
        rho: float = 6.0,
        beta_max: float = 0.8,
        corr_scale: float = 1e-2,
    ):
        der_K = jnp.ravel(der_K)
        if der_K.shape != (3,):
            raise ValueError(f"Expected der_K shape (3,), got {der_K.shape}")

        self.K_log = jnp.asarray(der_K)
        k_ss0, k_sb0, k_bb0 = der_K
        eps = 1e-6

        k_ss_base = k_ss0 / l_k
        k_bb_base = k_bb0 * l_k
        k_sb_base = k_sb0

        if k_ss_base <= 0 or k_bb_base <= 0:
            raise ValueError("Need positive base stiffness.")
        if k_ss_base * k_bb_base - 2.0 * k_sb_base**2 <= 0:
            raise ValueError("Base stiffness must be strictly PD.")

        s0 = k_bb_base
        r0 = k_ss_base / k_bb_base
        beta0 = jnp.sqrt(2.0) * k_sb_base / jnp.sqrt(k_ss_base * k_bb_base)

        log_r0 = jnp.log(r0)
        if jnp.abs(log_r0) >= rho:
            raise ValueError("Base ratio outside allowed range; increase rho or adjust l_k.")
        if jnp.abs(beta0) >= beta_max:
            raise ValueError("Coupling too large for beta_max.")

        raw_scale0 = inv_softplus(s0 - eps)
        raw_ratio0 = jnp.arctanh(jnp.clip(log_r0 / rho, -0.999999, 0.999999))
        raw_beta0 = jnp.arctanh(jnp.clip(beta0 / beta_max, -0.999999, 0.999999))

        self.theta = jnp.array([raw_scale0, raw_ratio0, raw_beta0])
        self.mlp = MLPTanhVector(5, 10, 3, key=key)
        self.rho = float(rho)
        self.beta_max = float(beta_max)
        self.corr_scale = float(corr_scale)
        self.l_k = float(l_k)

    def _bounded_parameters(self, del_strain: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        eps = 1e-6
        x = jnp.ravel(del_strain)
        dtheta = self.corr_scale * self.mlp(x)
        theta = self.theta + dtheta
        s = jax.nn.softplus(theta[0]) + eps
        log_r = self.rho * jnp.tanh(theta[1])
        r = jnp.exp(log_r)
        beta = self.beta_max * jnp.tanh(theta[2])
        return s, r, beta

    def get_K_entries(self, del_strain: Float[jax.Array, "5"]) -> jax.Array:
        s, r, beta = self._bounded_parameters(del_strain)
        k_bb_base = s
        k_ss_base = s * r
        k_sb_base = beta * s * jnp.sqrt(r / 2.0)
        return jnp.array(
            [self.l_k * k_ss_base, k_sb_base, k_bb_base / self.l_k]
        )

    def get_K_matrix(self, del_strain: Float[jax.Array, "5"]) -> jax.Array:
        k_ss, k_sb, k_bb = self.get_K_entries(del_strain)
        return jnp.array(
            [
                [k_ss, 0.0, k_sb],
                [0.0, k_ss, k_sb],
                [k_sb, k_sb, k_bb],
            ]
        )

    def __call__(self, del_strain: Float[jax.Array, "5"]) -> Float[jax.Array, ""]:
        del_strain = jnp.ravel(del_strain)
        e0, e1, eb = del_strain[0], del_strain[1], del_strain[3]
        k_ss, k_sb, k_bb = self.get_K_entries(del_strain)
        return (
            0.5 * k_ss * (e0**2 + e1**2)
            + k_sb * (e0 + e1) * eb
            + 0.5 * k_bb * eb**2
        )


# ---------------------------------------------------------------------------
# Notebook shorthand (optional)
# ---------------------------------------------------------------------------

Example2D = Triplet2DStiffnessMLP
Example2DKeps = Triplet2DKPairSoftplusMLP
ExampleMatrix = Triplet3DCholeskyMLP
ExampleCholesky = Triplet3DCholeskyConstant
ExampleSpectral = Triplet3DBoundedSpectral

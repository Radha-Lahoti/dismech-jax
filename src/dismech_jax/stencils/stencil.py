from abc import abstractmethod
from typing import Generic, Self, TypeVar

import jax
import jax.numpy as jnp
import equinox as eqx

from ..states import State

AuxT = TypeVar("AuxT", bound=State | None)


class Stencil(eqx.Module, Generic[AuxT]):
    """DDG stencils."""

    bar_strain: jax.Array

    @classmethod
    def init(cls, q: jax.Array, aux: AuxT, **kwargs) -> Self:
        """Get a Stencil instance with an initial configuration.

        Args:
            q (jax.Array): local DOFs.
            aux (State | None, optional): Aux variables. Defaults to None.

        Returns:
            Self: Stencil instance
        """
        temp_instance = cls(bar_strain=jnp.empty(0), **kwargs)
        bar_strain = temp_instance.get_strain(q, aux)
        return cls(bar_strain=bar_strain, **kwargs)

    def get_energy(self, q: jax.Array, model: eqx.Module, aux: AuxT) -> jax.Array:
        """Get scalar energy.

        Args:
            q (jax.Array): local DOFs.
            model (eqx.Module): Equinox model: `f(del_strain)-> scalar`.
            aux (State | None, optional): Aux variables. Defaults to None.

        Returns:
            jax.Array: Scalar energy.
        """
        del_strain = self.get_strain(q, aux) - self.bar_strain
        return model(del_strain)  # type: ignore

    @abstractmethod
    def get_strain(self, q: jax.Array, aux: AuxT) -> jax.Array:
        """Get strain vector.

        Args:
            q (jax.Array): local DOFs.
            aux (State | None, optional): Aux variables. Defaults to None.

        Returns:
            jax.Array: strain vector.
        """

    @staticmethod
    def get_epsilon(n0: jax.Array, n1: jax.Array, l_k: jax.Array) -> jax.Array:
        edge = n1 - n0
        edge_len = jnp.linalg.norm(edge)
        return edge_len / l_k - 1.0

    @staticmethod
    def get_kappa(
        n0: jax.Array,
        n1: jax.Array,
        n2: jax.Array,
        m1e: jax.Array,
        m2e: jax.Array,
        m1f: jax.Array,
        m2f: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        ee = n1 - n0
        ef = n2 - n1
        norm_e = jnp.linalg.norm(ee)
        norm_f = jnp.linalg.norm(ef)
        te = ee / norm_e
        tf = ef / norm_f
        chi = 1.0 + jnp.sum(te * tf)
        # jax.debug.print("chi = {}, small = {}", chi, chi < 1e-6)
        kb = 2.0 * jnp.cross(te, tf) / chi
        kappa1 = 0.5 * jnp.sum(kb * (m2e + m2f))
        kappa2 = -0.5 * jnp.sum(kb * (m1e + m1f))
        return kappa1, kappa2

    @staticmethod
    def get_tau(theta_e: jax.Array, theta_f: jax.Array, beta: jax.Array) -> jax.Array:
        return theta_f - theta_e + beta

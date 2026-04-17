from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax

from .bc import AbstractBC
from .energy import AbstractEnergy, Gravity

from ..models import DER
from ..stencils import Triplet, Triplet2D
from ..states import TripletState
from ..params import Geometry, Material
from .system import System
from ..solver import solve, solve_with_aux


class Rod(System[TripletState]):
    triplets: Triplet
    mass: jax.Array
    q0: jax.Array
    E_ext: AbstractEnergy
    bc: AbstractBC

    @classmethod
    def from_geometry(
        cls,
        geom: Geometry,
        material: Material,
        N: int = 30,
        bc: AbstractBC = AbstractBC(),
        origin: jax.Array = jnp.array([0.0, 0.0, 0.0]),
        gravity: float = -9.81,
        is_2d: bool = False,
    ) -> tuple["Rod", TripletState | None]:
        if N < 3:
            raise ValueError("Cannot create a rod with less than 3 nodes.")
        if geom.length < 1e-6:
            raise ValueError("Cannot create a rod less than 1 um.")

        q0 = jnp.zeros(4 * N - 1)
        xs = jnp.linspace(0, geom.length, N) + origin[0]
        q0 = q0.at[0::4].set(xs)
        q0 = q0.at[1::4].set(origin[1])
        q0 = q0.at[2::4].set(origin[2])
        batch_q = Rod._global_q_to_batch_q(q0)

        l_ks = jnp.diff(xs)
        mass = Rod._get_mass(geom, material, l_ks)

        N_triplets = batch_q.shape[0]
        batch_l_ks = jax.vmap(lambda i: jax.lax.dynamic_slice(l_ks, (i,), (2,)))(
            jnp.arange(N_triplets)
        )

        if is_2d:
            batch_aux = None
            triplets = jax.vmap(lambda q, l_k: Triplet2D.init(q, None, l_k=l_k))(
                batch_q, batch_l_ks
            )
        else:
            t_pair = jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            d1_pair = jnp.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
            ts = jnp.broadcast_to(t_pair, (N_triplets, 2, 3))
            d1s = jnp.broadcast_to(d1_pair, (N_triplets, 2, 3))
            betas = jnp.zeros(N_triplets)

            batch_aux = jax.vmap(TripletState)(ts, d1s, betas)
            triplets = jax.vmap(lambda q, a, l_k: Triplet.init(q, a, l_k=l_k))(
                batch_q, batch_aux, batch_l_ks
            )

        F_ext = jnp.zeros_like(q0).at[2::4].set(mass[2::4] * gravity)

        rod = Rod(
            triplets=triplets,
            E_ext=Gravity(F_ext),
            bc=bc,
            q0=q0,
            mass=mass,
        )
        return rod, batch_aux
    
    @classmethod
    def from_endpoints(
        cls,
        start: jax.Array,
        end: jax.Array,
        material: Material,
        N: int = 30,
        bc: AbstractBC = AbstractBC(),
        gravity: float = -9.81,
        is_2d: bool = False,
    ) -> tuple["Rod", TripletState | None]:

        if N < 3:
            raise ValueError("Cannot create a rod with less than 3 nodes.")

        start = jnp.asarray(start)
        end = jnp.asarray(end)

        length = jnp.linalg.norm(end - start)
        if length < 1e-6:
            raise ValueError("Start and end points are too close.")

        # ---------------------------------------
        # 1. Interpolate positions
        # ---------------------------------------
        alphas = jnp.linspace(0.0, 1.0, N)[:, None]  # (N,1)
        points = start + alphas * (end - start)      # (N,3)

        # ---------------------------------------
        # 2. Build q0
        # ---------------------------------------
        q0 = jnp.zeros(4 * N - 1)
        q0 = q0.at[0::4].set(points[:, 0])
        q0 = q0.at[1::4].set(points[:, 1])
        q0 = q0.at[2::4].set(points[:, 2])

        batch_q = Rod._global_q_to_batch_q(q0)

        # ---------------------------------------
        # 3. Segment lengths
        # ---------------------------------------
        l_ks = jnp.linalg.norm(points[1:] - points[:-1], axis=1)

        mass = Rod._get_mass(
        Geometry(length=length), material, l_ks
        )

        # ---------------------------------------
        # 4. Triplet construction
        # ---------------------------------------
        N_triplets = batch_q.shape[0]

        batch_l_ks = jax.vmap(
            lambda i: jax.lax.dynamic_slice(l_ks, (i,), (2,))
        )(jnp.arange(N_triplets))

        if is_2d:
            batch_aux = None
            triplets = jax.vmap(lambda q, l_k: Triplet2D.init(q, None, l_k=l_k))(
                batch_q, batch_l_ks
            )
        else:
            # Compute tangent direction
            direction = (end - start) / length

            # Construct orthonormal frame (simple version)
            # pick arbitrary perpendicular vector
            ref = jnp.array([0.0, 0.0, 1.0])
            d1 = jnp.cross(direction, ref)
            d1 = d1 / (jnp.linalg.norm(d1) + 1e-8)

            t_pair = jnp.stack([direction, direction])
            d1_pair = jnp.stack([d1, d1])

            ts = jnp.broadcast_to(t_pair, (N_triplets, 2, 3))
            d1s = jnp.broadcast_to(d1_pair, (N_triplets, 2, 3))
            betas = jnp.zeros(N_triplets)

            batch_aux = jax.vmap(TripletState)(ts, d1s, betas)

            triplets = jax.vmap(
                lambda q, a, l_k: Triplet.init(q, a, l_k=l_k)
            )(batch_q, batch_aux, batch_l_ks)

        # ---------------------------------------
        # 5. External force (gravity)
        # ---------------------------------------
        F_ext = jnp.zeros_like(q0).at[2::4].set(mass[2::4] * gravity)

        rod = Rod(
            triplets=triplets,
            E_ext=Gravity(F_ext),
            bc=bc,
            q0=q0,
            mass=mass,
        )

        return rod, batch_aux

    def with_bc(self, bc: AbstractBC) -> "Rod":
        return eqx.tree_at(lambda r: r.bc, self, bc)

    def get_DER(self, geom: Geometry, material: Material) -> DER:
        return DER.from_legacy(self.triplets.l_k[0, 0], geom, material)

    def get_q(self, _lambda: jax.Array, q0: jax.Array) -> jax.Array:
        return self.bc.apply(q0, _lambda)

    def get_E(
        self, _lambda: jax.Array, q: jax.Array, model: eqx.Module, aux: TripletState
    ) -> jax.Array:
        batch_qs = self._global_q_to_batch_q(q)
        E_int = jnp.sum(
            jax.vmap(lambda t, q_loc, _aux: t.get_energy(q_loc, model, _aux))(
                self.triplets, batch_qs, aux
            )
        )
        return E_int + self.E_ext(q, _lambda)

    def get_F(
        self, _lambda: jax.Array, q: jax.Array, model: eqx.Module, aux: TripletState
    ) -> jax.Array:
        mask = self.bc.mask(q)
        return mask * jax.grad(self.get_E, 1)(_lambda, q, model, aux)

    def get_H(
        self, _lambda: jax.Array, q: jax.Array, model: eqx.Module, aux: TripletState
    ) -> jax.Array:
        mask = self.bc.mask(q)
        H = jax.hessian(self.get_E, 1)(_lambda, q, model, aux)
        H = H * mask[:, None] * mask[None, :]
        diag_idx = jnp.arange(H.shape[0])
        return H.at[diag_idx, diag_idx].add(1.0 - mask)

    def get_strains(
        self,
        q: jax.Array,
        aux: TripletState,
    ) -> jax.Array:
        """
        Absolute strains for each triplet at a single configuration.

        Parameters
        ----------
        q : (dof,)
            Global rod DOFs for one configuration.
        aux : TripletState
            Aux state for one configuration, with leading dimension n_triplets.

        Returns
        -------
        strains : (n_triplets, n_strain)
            Absolute strain vector for each triplet.
        """
        batch_qs = self._global_q_to_batch_q(q)
        return jax.vmap(
            lambda t, q_loc, aux_loc: t.get_strain(q_loc, aux_loc)
        )(self.triplets, batch_qs, aux)

    def get_del_strains(
        self,
        q: jax.Array,
        aux: TripletState,
    ) -> jax.Array:
        """
        Reduced strains for each triplet at a single configuration:
            del_strain = strain - bar_strain
        """
        return self.get_strains(q, aux) - self.triplets.bar_strain
    
    def get_ode_term(self) -> diffrax.ODETerm:
        if self.is_batched(self.in_axes):
            raise NotImplementedError(
                f"get_ode_term: {self} contains a batched BC or E_ext. This is not supported yet!"
            )

        @eqx.filter_jit
        def rhs(_lambda, y, args):
            model, aux = args
            _lambda = jnp.asarray(_lambda)

            n_dofs = self.q0.shape[0]
            q, v = y[:n_dofs], y[n_dofs:]

            q_fixed, v_fixed = jax.jvp(
                lambda l: self.bc.apply(q, l), (_lambda,), (jnp.ones_like(_lambda),)
            )

            v = v * self.bc.mask(q) + v_fixed * (1.0 - self.bc.mask(q))
            a = -self.get_F(_lambda, q_fixed, model, aux) / self.mass
            return jnp.concatenate([v, a])

        return diffrax.ODETerm(rhs)

    def _infer_batch_size(self) -> int:
        """
        Infer batch size from batched BC or batched external energy.

        Supports:
        - BatchedLinearBC with xb_m / xb_c / idx_b
        - BatchedDirectBC with xb / idx_b
        """
        batch_size = None

        if self.bc is not None and self.bc.in_axes is not None:
            if getattr(self.bc.in_axes, "xb_m", None) == 0:
                batch_size = self.bc.xb_m.shape[0]
            elif getattr(self.bc.in_axes, "xb_c", None) == 0:
                batch_size = self.bc.xb_c.shape[0]
            elif getattr(self.bc.in_axes, "xb", None) == 0:
                batch_size = self.bc.xb.shape[0]
            elif getattr(self.bc.in_axes, "idx_b", None) == 0:
                batch_size = self.bc.idx_b.shape[0]

        if batch_size is None and self.E_ext is not None and self.E_ext.in_axes is not None:
            if getattr(self.E_ext.in_axes, "F_ext", None) == 0:
                batch_size = self.E_ext.F_ext.shape[0]

        if batch_size is None:
            raise ValueError("Rod is marked batched, but could not infer batch size.")

        return batch_size

    def _broadcast_q0_for_batch(self, batch_size: int) -> jax.Array:
        q0 = self.q0
        if q0.ndim == 1:
            q0 = jnp.broadcast_to(q0[None, :], (batch_size, q0.shape[0]))
        elif q0.ndim == 2:
            if q0.shape[0] != batch_size:
                raise ValueError(
                    f"Batched q0 has wrong batch size: q0.shape={q0.shape}, expected batch size {batch_size}"
                )
        else:
            raise ValueError(f"Expected q0 to have ndim 1 or 2, got shape {q0.shape}")
        return q0

    @eqx.filter_jit
    def solve(
        self,
        model: eqx.Module,
        lambdas: jax.Array,
        aux: TripletState,
        iters: int = 10,
        ls_steps: int = 10,
        c1: float = 1e-4,
        max_dlambda: float = 1e-1,
        abs_tol: float = 1e-8,
        rel_tol: float = 1e-6,
        fail_on_nonconvergence: bool = False,
    ) -> jax.Array:
        args = (
            model, lambdas, self.q0, aux, self,
            iters, ls_steps, c1, max_dlambda,
            abs_tol, rel_tol, fail_on_nonconvergence
        )

        if not self.is_batched(self.in_axes):
            return solve(*args)

        batch_size = self._infer_batch_size()
        q0 = self._broadcast_q0_for_batch(batch_size)

        return eqx.filter_vmap(
            solve,
            in_axes=(
                None, None, 0, None, self.in_axes,
                None, None, None, None, None, None, None
            ),
        )(
            model, lambdas, q0, aux, self,
            iters, ls_steps, c1, max_dlambda,
            abs_tol, rel_tol, fail_on_nonconvergence
        )

    @staticmethod
    def _global_q_to_batch_q(q: jax.Array) -> jax.Array:
        N_triplets = (q.shape[0] + 1) // 4 - 2
        starts = jnp.arange(N_triplets) * 4
        return jax.vmap(lambda s: jax.lax.dynamic_slice(q, (s,), (11,)))(starts)

    @staticmethod
    def _get_mass(geom: Geometry, material: Material, l_ks: jax.Array) -> jax.Array:
        N = l_ks.shape[0] + 1
        mass = jnp.zeros(N * 4 - 1)
        A = geom.axs if geom.axs else jnp.pi * geom.r0**2

        weights = 0.5 * l_ks[0]
        v_ref_len = jnp.ones(N) * 2 * weights
        v_ref_len = v_ref_len.at[0].set(weights)
        v_ref_len = v_ref_len.at[-1].set(weights)
        dm_nodes = v_ref_len * A * material.density
        node_start_indices = jnp.arange(N) * 4
        for i in range(3):
            mass = mass.at[node_start_indices + i].set(dm_nodes)

        factor = geom.jxs / geom.axs if geom.jxs and geom.axs else geom.r0**2 / 2
        dm_edges = l_ks * A * material.density * factor
        edge_indices = jnp.arange(N - 1) * 4 + 3
        mass = mass.at[edge_indices].set(dm_edges)
        # print("total mass:", jnp.sum(mass))
        return mass

    @property
    def in_axes(self) -> "Rod":
        return Rod(
            triplets=None,  # type: ignore
            mass=None,  # type: ignore
            q0=None,  # type: ignore
            E_ext=self.E_ext.in_axes,  # type: ignore
            bc=self.bc.in_axes,  # type: ignore
        )

    @staticmethod
    def is_batched(axes) -> bool:
        leaves = jax.tree_util.tree_leaves(axes)
        return any(leaf is not None for leaf in leaves)

    @eqx.filter_jit
    def solve_with_aux(
        self,
        model: eqx.Module,
        lambdas: jax.Array,
        aux: TripletState,
        iters: int = 10,
        ls_steps: int = 10,
        c1: float = 1e-4,
        max_dlambda: float = 1e-1,
        abs_tol: float = 1e-8,
        rel_tol: float = 1e-6,
        fail_on_nonconvergence: bool = False,
    ):
        args = (
            model, lambdas, self.q0, aux, self,
            iters, ls_steps, c1, max_dlambda,
            abs_tol, rel_tol, fail_on_nonconvergence
        )

        if not self.is_batched(self.in_axes):
            return solve_with_aux(*args)

        batch_size = self._infer_batch_size()
        q0 = self._broadcast_q0_for_batch(batch_size)

        return eqx.filter_vmap(
            solve_with_aux,
            in_axes=(
                None, None, 0, None, self.in_axes,
                None, None, None, None, None, None, None
            ),
        )(
            model, lambdas, q0, aux, self,
            iters, ls_steps, c1, max_dlambda,
            abs_tol, rel_tol, fail_on_nonconvergence
        )

    def get_del_strain_history(
        self,
        qs: jax.Array,
        auxs: TripletState,
    ) -> jax.Array:
        """
        Reduced strain history.

        Parameters
        ----------
        qs : (T, dof) or (B, T, dof)
        auxs : TripletState with matching leading time dims
            For single trajectory:
                auxs.t.shape    = (T, n_triplets, 2, 3)
                auxs.d1.shape   = (T, n_triplets, 2, 3)
                auxs.beta.shape = (T, n_triplets)
            For batched trajectories:
                auxs has leading batch dimension as well.

        Returns
        -------
        del_strains : (T, n_triplets, n_strain) or (B, T, n_triplets, n_strain)
        """
        if qs.ndim == 3:
            return jax.vmap(self.get_del_strain_history)(qs, auxs)

        return jax.vmap(self.get_del_strains)(qs, auxs)
    



    # #### Solver with info (residual norms, convergence flags, etc.) ####
    # @eqx.filter_jit
    # def solve_with_info(
    #     self,
    #     model: eqx.Module,
    #     lambdas: jax.Array,
    #     aux: TripletState,
    #     iters: int = 10,
    #     ls_steps: int = 10,
    #     c1: float = 1e-4,
    #     max_dlambda: float = 1e-1,
    #     abs_tol: float = 1e-8,
    #     rel_tol: float = 1e-6,
    #     fail_on_nonconvergence: bool = False,
    # ):
    #     args = (
    #         model,
    #         lambdas,
    #         self.q0,
    #         aux,
    #         self,
    #         iters,
    #         ls_steps,
    #         c1,
    #         max_dlambda,
    #         abs_tol,
    #         rel_tol,
    #         fail_on_nonconvergence,
    #     )

    #     if not self.is_batched(self.in_axes):
    #         return solve_with_info(*args)

    #     batch_size = self._infer_batch_size()
    #     q0 = self._broadcast_q0_for_batch(batch_size)

    #     return eqx.filter_vmap(
    #         solve_with_info,
    #         in_axes=(
    #             None,   # model
    #             None,   # lambdas
    #             0,      # q0
    #             None,   # aux
    #             self.in_axes,
    #             None,   # iters
    #             None,   # ls_steps
    #             None,   # c1
    #             None,   # max_dlambda
    #             None,   # abs_tol
    #             None,   # rel_tol
    #             None,   # fail_on_nonconvergence
    #         ),
    #     )(
    #         model,
    #         lambdas,
    #         q0,
    #         aux,
    #         self,
    #         iters,
    #         ls_steps,
    #         c1,
    #         max_dlambda,
    #         abs_tol,
    #         rel_tol,
    #         fail_on_nonconvergence,
    #     )
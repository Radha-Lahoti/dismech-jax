import jax
import jax.numpy as jnp
import equinox as eqx

from .states import State
from .systems import System


def update_aux_state(
    aux: State,
    q: jax.Array,
    sys: System,
) -> State:
    if aux is None:
        return aux

    if hasattr(sys, "_global_q_to_batch_q"):
        batch_q = sys._global_q_to_batch_q(q)
        return jax.vmap(lambda a, q_loc: a.update(q_loc))(aux, batch_q)

    return jax.vmap(lambda a: a.update(q))(aux)


def compute_ift_gradient(
    _lambda: jax.Array,
    q_star: jax.Array,
    grad_obj: jax.Array,
    model: eqx.Module,
    aux: State,
    sys: System,
) -> eqx.Module:
    H = sys.get_H(_lambda, q_star, model, aux)
    H_reg = H.at[jnp.diag_indices(H.shape[0])].add(1e-8)
    v = jnp.linalg.solve(H_reg, grad_obj)
    _, vjp_fn = jax.vjp(lambda _m: sys.get_F(_lambda, q_star, _m, aux), model)
    (grads,) = vjp_fn(-v)
    return grads


@eqx.filter_custom_vjp
@eqx.filter_jit
def solve_step(
    model: eqx.Module,
    _lambda: jax.Array,
    q0: jax.Array,
    aux: State,
    sys: System,
    iters: int = 10,
    ls_steps: int = 10,
    c1: float = 1e-4,
    abs_tol: float = 1e-8,
    rel_tol: float = 1e-6,
    fail_on_nonconvergence: bool = False,
) -> jax.Array:
    """
    Fixed-length Newton solve for one continuation/load step.

    Added minimal convergence check:
      - absolute residual tolerance
      - relative residual reduction tolerance
      - optional hard failure via eqx.error_if

    This does NOT change the number of Newton iterations.
    """
    alphas = 0.5 ** jnp.arange(ls_steps)

    q_init = sys.get_q(_lambda, q0)
    aux_init = update_aux_state(aux, q_init, sys)

    def newton_step(carry, _):
        q, aux_cur, e_old, res = carry

        # Newton system
        H = sys.get_H(_lambda, q, model, aux_cur)
        H_reg = H.at[jnp.diag_indices(H.shape[0])].add(1e-8)
        delta_q = jnp.linalg.solve(H_reg, res)
        slope = jnp.dot(res, delta_q)

        # Parallel line search
        test_qs = q + alphas[:, None] * delta_q
        test_energies = jax.vmap(lambda _q: sys.get_E(_lambda, _q, model, aux_cur))(test_qs)

        # Since res = -grad(E), the directional derivative is grad(E)^T dq = -res^T dq.
        # Armijo therefore uses a decrease bound with a minus sign here.
        is_good = test_energies <= e_old - c1 * alphas * slope

        # If Armijo fails, take the smallest possible step
        safe_idx = jnp.where(jnp.any(is_good), jnp.argmax(is_good), ls_steps - 1)

        next_q = test_qs[safe_idx]
        next_e = test_energies[safe_idx]
        next_aux = update_aux_state(aux_cur, next_q, sys)
        next_res = -sys.get_F(_lambda, next_q, model, next_aux)

        return (next_q, next_aux, next_e, next_res), None

    init_e = sys.get_E(_lambda, q_init, model, aux_init)
    init_res = -sys.get_F(_lambda, q_init, model, aux_init)
    init_res_norm = jnp.linalg.norm(init_res)

    (final_q, _, _, final_res), _ = jax.lax.scan(
        newton_step, (q_init, aux_init, init_e, init_res), None, iters
    )

    final_res_norm = jnp.linalg.norm(final_res)
    rel_res_norm = final_res_norm / jnp.maximum(init_res_norm, 1e-16)

    converged = jnp.logical_or(final_res_norm < abs_tol, rel_res_norm < rel_tol)

    if fail_on_nonconvergence:
        final_q = eqx.error_if(
            final_q,
            ~converged,
            ("Newton solve did not converge. Residual norm: {}", final_res_norm)
        )

    return final_q


@solve_step.def_fwd
def solve_step_fwd(
    perturbed: eqx.Module,
    model: eqx.Module,
    _lambda: jax.Array,
    q0: jax.Array,
    aux: State,
    sys: System,
    iters: int = 10,
    ls_steps: int = 10,
    c1: float = 1e-4,
    abs_tol: float = 1e-8,
    rel_tol: float = 1e-6,
    fail_on_nonconvergence: bool = False,
) -> tuple[jax.Array, jax.Array]:
    final_q = solve_step(
        model, _lambda, q0, aux, sys,
        iters, ls_steps, c1,
        abs_tol, rel_tol, fail_on_nonconvergence
    )
    return final_q, final_q


@solve_step.def_bwd
def solve_step_bwd(
    res: jax.Array,
    grad_obj: jax.Array,
    perturbed: eqx.Module,
    model: eqx.Module,
    _lambda: jax.Array,
    q0: jax.Array,
    aux: State,
    sys: System,
    iters: int = 10,
    ls_steps: int = 10,
    c1: float = 1e-4,
    abs_tol: float = 1e-8,
    rel_tol: float = 1e-6,
    fail_on_nonconvergence: bool = False,
) -> eqx.Module:
    return compute_ift_gradient(_lambda, res, grad_obj, model, aux, sys)


@eqx.filter_custom_vjp
@eqx.filter_jit
def solve(
    model: eqx.Module,
    lambdas: jax.Array,
    q0: jax.Array,
    aux: State,
    sys: System,
    iters: int = 10,
    ls_steps: int = 10,
    c1: float = 1e-4,
    max_dt: float = 1e-1,
    abs_tol: float = 1e-8,
    rel_tol: float = 1e-6,
    fail_on_nonconvergence: bool = False,
) -> jax.Array:
    def scan_fn(res: tuple[jax.Array, State, jax.Array], target_lambda: jax.Array):
        _q, _aux, _current_lambda = res

        def cond_fn(val: tuple[jax.Array, State, jax.Array]):
            _, _, curr_L = val
            return curr_L < target_lambda

        def body_fn(carry: tuple[jax.Array, State, jax.Array]):
            q, aux, curr_L = carry
            next_L = jnp.minimum(curr_L + max_dt, target_lambda)
            new_q = solve_step(
                model, next_L, q, aux, sys,
                iters, ls_steps, c1,
                abs_tol, rel_tol, fail_on_nonconvergence
            )
            new_aux = update_aux_state(aux, new_q, sys)
            return new_q, new_aux, next_L

        final_q, final_aux, final_L = jax.lax.while_loop(
            cond_fn, body_fn, (_q, _aux, _current_lambda)
        )
        return (final_q, final_aux, final_L), final_q

    q_start = solve_step(
        model, lambdas[0], q0, aux, sys,
        iters, ls_steps, c1,
        abs_tol, rel_tol, fail_on_nonconvergence
    )
    aux_start = update_aux_state(aux, q_start, sys)
    _, qs = jax.lax.scan(scan_fn, (q_start, aux_start, lambdas[0]), lambdas)
    return qs


@solve.def_fwd
def solve_fwd(
    perturbed: eqx.Module,
    model: eqx.Module,
    lambdas: jax.Array,
    q0: jax.Array,
    aux: State,
    sys: System,
    iters: int = 10,
    ls_steps: int = 10,
    c1: float = 1e-4,
    max_dt: float = 1e-1,
    abs_tol: float = 1e-8,
    rel_tol: float = 1e-6,
    fail_on_nonconvergence: bool = False,
) -> tuple[jax.Array, tuple[jax.Array, State]]:
    def scan_fwd_fn(res: tuple[jax.Array, State, jax.Array], target_lambda: jax.Array):
        _q, _aux, _current_lambda = res

        def cond_fn(val):
            _, _, curr_L = val
            return curr_L < target_lambda

        def body_fn(val):
            q, aux, curr_L = val
            next_L = jnp.minimum(curr_L + max_dt, target_lambda)
            new_q = solve_step(
                model, next_L, q, aux, sys,
                iters, ls_steps, c1,
                abs_tol, rel_tol, fail_on_nonconvergence
            )
            new_aux = update_aux_state(aux, new_q, sys)
            return new_q, new_aux, next_L

        final_q, final_aux, final_L = jax.lax.while_loop(
            cond_fn, body_fn, (_q, _aux, _current_lambda)
        )
        return (final_q, final_aux, final_L), (final_q, _aux)

    q_start = solve_step(
        model, lambdas[0], q0, aux, sys,
        iters, ls_steps, c1,
        abs_tol, rel_tol, fail_on_nonconvergence
    )
    aux_start = update_aux_state(aux, q_start, sys)

    _, (qs, auxs) = jax.lax.scan(
        scan_fwd_fn, (q_start, aux_start, lambdas[0]), lambdas
    )
    return qs, (qs, auxs)


@solve.def_bwd
def solve_bwd(
    res: tuple[jax.Array, State],
    grad_obj: jax.Array,
    perturbed: eqx.Module,
    model: eqx.Module,
    lambdas: jax.Array,
    q0: jax.Array,
    aux: State,
    sys: System,
    iters: int = 10,
    ls_steps: int = 10,
    c1: float = 1e-4,
    max_dt: float = 1e-1,
    abs_tol: float = 1e-8,
    rel_tol: float = 1e-6,
    fail_on_nonconvergence: bool = False,
) -> eqx.Module:
    qs, auxs = res
    batched_ift_fn = jax.vmap(compute_ift_gradient, in_axes=(0, 0, 0, None, 0, None))
    batched_grads = batched_ift_fn(lambdas, qs, grad_obj, model, auxs, sys)
    total_grad = jax.tree.map(lambda x: jnp.sum(x, axis=0), batched_grads)
    return total_grad


### for debugging: return strains too
@eqx.filter_jit
def solve_with_aux(
    model,
    lambdas,
    q0,
    aux,
    sys,
    iters=10,
    ls_steps=10,
    c1=1e-4,
    max_dt=1e-1,
    abs_tol: float = 1e-8,
    rel_tol: float = 1e-6,
    fail_on_nonconvergence: bool = False,
):
    def scan_fn(res, target_lambda):
        _q, _aux, _current_lambda = res

        def cond_fn(val):
            _, _, curr_L = val
            return curr_L < target_lambda

        def body_fn(carry):
            q, aux, curr_L = carry
            next_L = jnp.minimum(curr_L + max_dt, target_lambda)
            new_q = solve_step(
                model, next_L, q, aux, sys,
                iters, ls_steps, c1,
                abs_tol, rel_tol, fail_on_nonconvergence
            )
            new_aux = update_aux_state(aux, new_q, sys)
            return new_q, new_aux, next_L

        final_q, final_aux, final_L = jax.lax.while_loop(
            cond_fn, body_fn, (_q, _aux, _current_lambda)
        )
        return (final_q, final_aux, final_L), (final_q, final_aux)

    q_start = solve_step(
        model, lambdas[0], q0, aux, sys,
        iters, ls_steps, c1,
        abs_tol, rel_tol, fail_on_nonconvergence
    )
    aux_start = update_aux_state(aux, q_start, sys)

    _, (qs, auxs) = jax.lax.scan(
        scan_fn, (q_start, aux_start, lambdas[0]), lambdas
    )
    return qs, auxs

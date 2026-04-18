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


# =========================================================
# IFT gradient
# =========================================================
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


# =========================================================
# One Newton solve step with diagnostics
# =========================================================
@eqx.filter_jit
def solve_step_with_info(
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
):
    """
    Fixed-length Newton solve with residual tracking.

    Returns
    -------
    final_q : jax.Array
        Final Newton iterate.
    res_hist : jax.Array, shape (iters + 1,)
        Residual norm history, including the initial residual.
    final_res_norm : jax.Array, scalar
        Final residual norm.
    converged : jax.Array, bool scalar
        Whether final residual satisfies abs_tol or rel_tol reduction.
    """
    alphas = 0.5 ** jnp.arange(ls_steps)

    q_init = sys.get_q(_lambda, q0)
    aux_init = update_aux_state(aux, q_init, sys)

    def newton_step(carry, _):
        q, aux_cur, e_old, res = carry

        # Hessian and regularized Newton direction
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
        next_res_norm = jnp.linalg.norm(next_res)

        return (next_q, next_aux, next_e, next_res), next_res_norm

    # Initial iterate at this lambda
    init_e = sys.get_E(_lambda, q_init, model, aux_init)
    init_res = -sys.get_F(_lambda, q_init, model, aux_init)
    init_res_norm = jnp.linalg.norm(init_res)

    # Fixed number of Newton iterations
    (final_q, _, _, final_res), step_res_hist = jax.lax.scan(
        newton_step, (q_init, aux_init, init_e, init_res), None, iters
    )

    final_res_norm = jnp.linalg.norm(final_res)

    # Residual history includes the initial residual norm
    res_hist = jnp.concatenate([init_res_norm[None], step_res_hist], axis=0)

    # Convergence test:
    # 1) absolute residual below abs_tol
    # 2) or relative reduction below rel_tol
    denom = jnp.maximum(init_res_norm, 1e-16)
    rel_res = final_res_norm / denom
    converged = jnp.logical_or(final_res_norm < abs_tol, rel_res < rel_tol)

    if fail_on_nonconvergence:
        final_q = eqx.error_if(
            final_q,
            ~converged,
            "solve_step_with_info: Newton solve did not converge within fixed iterations.",
        )

    return final_q, res_hist, final_res_norm, converged


# =========================================================
# Original solve_step API preserved
# =========================================================
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
) -> jax.Array:
    final_q, _, _, _ = solve_step_with_info(
        model=model,
        _lambda=_lambda,
        q0=q0,
        aux=aux,
        sys=sys,
        iters=iters,
        ls_steps=ls_steps,
        c1=c1,
        abs_tol=1e-8,
        rel_tol=1e-6,
        fail_on_nonconvergence=False,
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
) -> tuple[jax.Array, jax.Array]:
    final_q = solve_step(model, _lambda, q0, aux, sys, iters, ls_steps, c1)
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
) -> eqx.Module:
    return compute_ift_gradient(_lambda, res, grad_obj, model, aux, sys)


# =========================================================
# Full continuation solve with diagnostics
# =========================================================
@eqx.filter_jit
def solve_with_info(
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
):
    """
    Fixed-length continuation solve with diagnostics.

    Returns
    -------
    qs : jax.Array
        Solution at each target lambda, shape (n_lambda, dof)

    info : dict
        Contains:
        - "res_hist": residual histories for the final substep used to reach each target lambda
                      shape (n_lambda, iters + 1)
        - "final_res_norm": final residual norm for that final substep, shape (n_lambda,)
        - "converged": convergence flag for that final substep, shape (n_lambda,)
        - "failed": logical negation of converged, shape (n_lambda,)
    """

    def scan_fn(res, target_lambda):
        _q, _aux, _current_lambda = res

        # Default placeholders in case while_loop does not run
        dummy_hist = jnp.zeros((iters + 1,), dtype=_q.dtype)
        dummy_res_norm = jnp.array(0.0, dtype=_q.dtype)
        dummy_conv = jnp.array(True)

        def cond_fn(val):
            _, _, curr_L, _, _, _ = val
            return curr_L < target_lambda

        def body_fn(carry):
            q, aux, curr_L, _, _, _ = carry
            next_L = jnp.minimum(curr_L + max_dt, target_lambda)

            new_q, res_hist, final_res_norm, converged = solve_step_with_info(
                model=model,
                _lambda=next_L,
                q0=q,
                aux=aux,
                sys=sys,
                iters=iters,
                ls_steps=ls_steps,
                c1=c1,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
                fail_on_nonconvergence=fail_on_nonconvergence,
            )

            new_aux = update_aux_state(aux, new_q, sys)
            return new_q, new_aux, next_L, res_hist, final_res_norm, converged

        final_q, final_aux, final_L, last_hist, last_res_norm, last_conv = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (_q, _aux, _current_lambda, dummy_hist, dummy_res_norm, dummy_conv),
        )

        return (
            final_q,
            final_aux,
            final_L,
        ), (
            final_q,
            last_hist,
            last_res_norm,
            last_conv,
        )

    # First lambda solve
    q_start, hist_start, resnorm_start, conv_start = solve_step_with_info(
        model=model,
        _lambda=lambdas[0],
        q0=q0,
        aux=aux,
        sys=sys,
        iters=iters,
        ls_steps=ls_steps,
        c1=c1,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        fail_on_nonconvergence=fail_on_nonconvergence,
    )
    aux_start = update_aux_state(aux, q_start, sys)

    _, (qs, res_hists, final_res_norms, convergeds) = jax.lax.scan(
        scan_fn,
        (q_start, aux_start, lambdas[0]),
        lambdas,
    )

    # Make first entry reflect the explicit initial solve
    res_hists = res_hists.at[0].set(hist_start)
    final_res_norms = final_res_norms.at[0].set(resnorm_start)
    convergeds = convergeds.at[0].set(conv_start)

    info = {
        "res_hist": res_hists,                # (n_lambda, iters + 1)
        "final_res_norm": final_res_norms,    # (n_lambda,)
        "converged": convergeds,              # (n_lambda,)
        "failed": ~convergeds,                # (n_lambda,)
    }
    return qs, info


# =========================================================
# Original solve API preserved
# =========================================================
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
) -> jax.Array:
    qs, _ = solve_with_info(
        model=model,
        lambdas=lambdas,
        q0=q0,
        aux=aux,
        sys=sys,
        iters=iters,
        ls_steps=ls_steps,
        c1=c1,
        max_dt=max_dt,
        abs_tol=1e-8,
        rel_tol=1e-6,
        fail_on_nonconvergence=False,
    )
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
) -> tuple[jax.Array, tuple[jax.Array, State]]:
    def scan_fwd_fn(res, target_lambda):
        _q, _aux, _current_lambda = res

        def cond_fn(val):
            _, _, curr_L = val
            return curr_L < target_lambda

        def body_fn(val):
            q, aux, curr_L = val
            next_L = jnp.minimum(curr_L + max_dt, target_lambda)
            new_q = solve_step(model, next_L, q, aux, sys, iters, ls_steps, c1)
            new_aux = update_aux_state(aux, new_q, sys)
            return new_q, new_aux, next_L

        final_q, final_aux, final_L = jax.lax.while_loop(
            cond_fn, body_fn, (_q, _aux, _current_lambda)
        )
        return (final_q, final_aux, final_L), (final_q, _aux)

    q_start = solve_step(model, lambdas[0], q0, aux, sys, iters, ls_steps, c1)
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
) -> eqx.Module:
    qs, auxs = res
    batched_ift_fn = jax.vmap(compute_ift_gradient, in_axes=(0, 0, 0, None, 0, None))
    batched_grads = batched_ift_fn(lambdas, qs, grad_obj, model, auxs, sys)
    total_grad = jax.tree.map(lambda x: jnp.sum(x, axis=0), batched_grads)
    return total_grad


# =========================================================
# Debugging solve: return aux + diagnostics
# =========================================================
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
    abs_tol=1e-8,
    rel_tol=1e-6,
    fail_on_nonconvergence=False,
):
    def scan_fn(res, target_lambda):
        _q, _aux, _current_lambda = res

        dummy_hist = jnp.zeros((iters + 1,), dtype=_q.dtype)
        dummy_res_norm = jnp.array(0.0, dtype=_q.dtype)
        dummy_conv = jnp.array(True)

        def cond_fn(val):
            _, _, curr_L, _, _, _ = val
            return curr_L < target_lambda

        def body_fn(carry):
            q, aux, curr_L, _, _, _ = carry
            next_L = jnp.minimum(curr_L + max_dt, target_lambda)

            new_q, res_hist, final_res_norm, converged = solve_step_with_info(
                model=model,
                _lambda=next_L,
                q0=q,
                aux=aux,
                sys=sys,
                iters=iters,
                ls_steps=ls_steps,
                c1=c1,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
                fail_on_nonconvergence=fail_on_nonconvergence,
            )

            new_aux = update_aux_state(aux, new_q, sys)
            return new_q, new_aux, next_L, res_hist, final_res_norm, converged

        final_q, final_aux, final_L, last_hist, last_res_norm, last_conv = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (_q, _aux, _current_lambda, dummy_hist, dummy_res_norm, dummy_conv),
        )

        return (
            final_q,
            final_aux,
            final_L,
        ), (
            final_q,
            final_aux,
            last_hist,
            last_res_norm,
            last_conv,
        )

    # Initial solve
    q_start, hist_start, resnorm_start, conv_start = solve_step_with_info(
        model=model,
        _lambda=lambdas[0],
        q0=q0,
        aux=aux,
        sys=sys,
        iters=iters,
        ls_steps=ls_steps,
        c1=c1,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        fail_on_nonconvergence=fail_on_nonconvergence,
    )
    aux_start = update_aux_state(aux, q_start, sys)

    _, (qs, auxs, res_hists, final_res_norms, convergeds) = jax.lax.scan(
        scan_fn,
        (q_start, aux_start, lambdas[0]),
        lambdas,
    )

    # Fix first entry explicitly
    res_hists = res_hists.at[0].set(hist_start)
    final_res_norms = final_res_norms.at[0].set(resnorm_start)
    convergeds = convergeds.at[0].set(conv_start)

    info = {
        "res_hist": res_hists,                # (n_lambda, iters + 1)
        "final_res_norm": final_res_norms,    # (n_lambda,)
        "converged": convergeds,              # (n_lambda,)
        "failed": ~convergeds,                # (n_lambda,)
    }

    return qs, auxs, info

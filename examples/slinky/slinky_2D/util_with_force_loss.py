import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import dismech_jax as djx
from dismech_jax.solver import update_aux_state

from Energy_NN_architectures import ModelParams


# =========================================================
# Dataset (ONLY direct BC, with optional measured reaction force)
# =========================================================
class Dataset(eqx.Module):
    qs: jax.Array        # (n_traj, T, dof)
    xb: jax.Array        # (n_traj, T, n_b)
    idx_b: jax.Array     # (n_b,) or (n_traj, n_b)
    lambdas: jax.Array   # (n_traj, T) or (T,)
    valid: jax.Array     # (n_traj, T)
    forces: jax.Array | None = None  # (n_traj, T), (n_traj, T, k), or None

    @staticmethod
    def load(path, force_key=None):
        data = np.load(path)
        forces = None
        if force_key is not None:
            forces = jnp.asarray(data[force_key])
        else:
            for key in ("F", "forces", "force", "reaction_forces", "reaction_force"):
                if key in data.files:
                    forces = jnp.asarray(data[key])
                    break
        return Dataset(
            qs=jnp.asarray(data["qs"]),
            xb=jnp.asarray(data["xb"]),
            idx_b=jnp.asarray(data["idx_b"]),
            lambdas=jnp.asarray(data["lambdas"]),
            valid=jnp.asarray(data["valid"]),
            forces=forces,
        )


def validate_dataset_compatibility(base, data: Dataset, label: str):
    if data.qs.ndim != 3:
        raise ValueError(f"{label} dataset qs must have shape (n_traj, T, dof), got {data.qs.shape}.")
    if data.xb.ndim != 3:
        raise ValueError(f"{label} dataset xb must have shape (n_traj, T, n_b), got {data.xb.shape}.")
    if data.xb.shape[:2] != data.qs.shape[:2]:
        raise ValueError(
            f"{label} dataset xb leading shape {data.xb.shape[:2]} must match "
            f"qs leading shape {data.qs.shape[:2]}."
        )
    if data.valid.shape != data.qs.shape[:2]:
        raise ValueError(
            f"{label} dataset valid shape {data.valid.shape} must match "
            f"qs leading shape {data.qs.shape[:2]}."
        )
    if data.lambdas.ndim == 1:
        if data.lambdas.shape[0] != data.qs.shape[1]:
            raise ValueError(
                f"{label} dataset lambdas shape {data.lambdas.shape} must have length T={data.qs.shape[1]}."
            )
    elif data.lambdas.shape != data.qs.shape[:2]:
        raise ValueError(
            f"{label} dataset lambdas shape {data.lambdas.shape} must be either "
            f"(T,) or match qs leading shape {data.qs.shape[:2]}."
        )

    expected_dof = base.q0.shape[0]
    actual_dof = data.qs.shape[-1]
    if actual_dof != expected_dof:
        inferred_n = (actual_dof + 1) // 4
        base_n = (expected_dof + 1) // 4
        raise ValueError(
            f"{label} dataset dof ({actual_dof}) does not match base rod dof ({expected_dof}). "
            f"This dataset appears to require N={inferred_n} nodes, but the current properties use N={base_n}."
        )
    if data.forces is not None and data.forces.shape[:2] != data.qs.shape[:2]:
        raise ValueError(
            f"{label} force data leading shape {data.forces.shape[:2]} must match "
            f"qs leading shape {data.qs.shape[:2]}."
        )

# =========================================================
# Base slinky rod (fixed)
# =========================================================
def get_slinky(properties):
    geom = djx.Geometry(properties.length, properties.r0)
    mat = djx.Material(properties.density, properties.E)

    if properties.start is not None and properties.end is not None:
        rod, aux = djx.Rod.from_endpoints(
            start=properties.start,
            end=properties.end,
            material=mat,
            N=properties.N,
        )
    else:
        rod, aux = djx.Rod.from_geometry(geom, mat, N=properties.N)

    if properties.mass is not None:
        mass = properties.mass
        N = properties.N
        assert N >= 2

        g = -9.81
        f = jnp.zeros(4 * N - 1)

        # set all node z entries as interior-node weights first
        f = f.at[2::4].set(mass / (N - 1) * g)

        # correct the two end nodes to half-weight
        f = f.at[2].set(mass / (2 * (N - 1)) * g)      # first node z
        f = f.at[4 * N - 2].set(mass / (2 * (N - 1)) * g)  # last node z
        rod = eqx.tree_at(lambda r: r.E_ext, rod, djx.Gravity(f))

    return rod, aux


# =========================================================
# Predict
# =========================================================
def predict(
    model,
    base,
    aux,
    idx_b,
    xb,
    lambdas,
    max_dlambda=5e-3,
    iters=5,
    ls_steps=10,
    abs_tol=1e-8,
    rel_tol=1e-6,
    fail_on_nonconvergence=False,
    early_stop=False,
):
    n_traj = xb.shape[0]

    # handle shared vs per-trajectory lambdas
    if lambdas.ndim == 1:
        lam_all = jnp.broadcast_to(lambdas, (n_traj, lambdas.shape[0]))
    else:
        lam_all = lambdas

    # handle shared vs per-trajectory idx_b
    if idx_b.ndim == 1:
        idx_all = jnp.broadcast_to(idx_b, (n_traj, idx_b.shape[0]))
    else:
        idx_all = idx_b

    def predict_one(ib, xb_i, lam_i):
        bc = djx.DirectBC(idx_b=ib, xb=xb_i, lambdas=lam_i)
        rod = base.with_bc(bc)
        return rod.solve(
            model,
            lam_i,
            aux,
            max_dlambda=max_dlambda,
            iters=iters,
            ls_steps=ls_steps,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            fail_on_nonconvergence=fail_on_nonconvergence,
            early_stop=early_stop,
        )

    pred = jax.vmap(predict_one)(idx_all, xb, lam_all)
    return pred

# =========================================================
# Hessian regularization
# =========================================================
def energy_hessian_spectral_regularizer(
    model,
    rod,
    aux,
    lambdas,
    qs,
    valid,
    key,
    n_probes=1,
):
    """
    Hutchinson/Frobenius proxy for the spectral size of the energy Hessian.

    For the Newton solve, the relevant local linear operator is
        H(q*) = d^2 E / dq^2,
    evaluated at the solved equilibrium q*. Following the DEQ Jacobian
    regularizer, we avoid eigendecompositions and use
        rho(H)^2 <= ||H||_2^2 <= ||H||_F^2
    with the unbiased Hutchinson estimate
        ||H||_F^2 = E_v ||H v||^2,  v ~ N(0, I).
    The division by the DOF count matches the DEQ paper's normalization by d.
    """
    keys = jax.random.split(key, qs.shape[0])
    dof = qs.shape[-1]

    def penalty_one(_lambda, q, is_valid, sample_key):
        aux_q = update_aux_state(aux, q, rod)
        H = rod.get_H(_lambda, q, model, aux_q)
        H = 0.5 * (H + H.T)
        free_mask = rod.bc.mask(q)
        H = H * free_mask[:, None] * free_mask[None, :]
        free_dof = jnp.maximum(jnp.sum(free_mask), 1.0)

        probes = jax.random.normal(sample_key, (n_probes, dof), dtype=q.dtype)
        h_probes = jax.vmap(lambda v: H @ v)(probes)
        estimate = jnp.mean(jnp.sum(h_probes**2, axis=-1)) / free_dof
        return jnp.where(is_valid, estimate, 0.0)

    penalties = jax.vmap(penalty_one)(lambdas, qs, valid, keys)
    return jnp.sum(penalties) / jnp.maximum(jnp.sum(valid), 1.0)


# =========================================================
# Reaction-force loss helpers
# =========================================================
class _AllFreeBC(djx.AbstractBC):
    """Identity BC used to expose constrained DOF forces from Rod.get_F."""

    def apply(self, q: jax.Array, _lambda: jax.Array) -> jax.Array:
        return q

    def mask(self, q: jax.Array) -> jax.Array:
        return jnp.ones_like(q)


def _first_boundary_node_force_indices(
    idx_b,
    q_size,
    force_components=(0, 1, 2),
):
    del q_size
    idx_b = jnp.asarray(idx_b)
    node_ids = jnp.where(jnp.mod(idx_b, 4) < 3, idx_b // 4, -1)
    valid_node_ids = jnp.where(node_ids >= 0, node_ids, jnp.iinfo(idx_b.dtype).max)
    first_node = jnp.min(valid_node_ids)
    force_components = jnp.asarray(force_components)
    return 4 * first_node + force_components


def predict_reaction_force(
    model,
    base,
    aux,
    lambdas,
    qs_true,
    idx_b,
    valid,
    force_components=(0, 1, 2),
    force_sign=1.0,
):
    """
    Learned reaction force at the first constrained translational node.

    Rod.get_F masks constrained DOFs for Newton solves. To recover reaction
    forces on prescribed DOFs, evaluate the same learned energy with an
    all-free BC and the measured configuration q_true.
    """
    force_idx = _first_boundary_node_force_indices(
        idx_b,
        qs_true.shape[-1],
        force_components=force_components,
    )
    force_rod = base.with_bc(_AllFreeBC())

    def force_one(_lambda, q, is_valid):
        aux_q = update_aux_state(aux, q, force_rod)
        force = force_sign * force_rod.get_F(_lambda, q, model, aux_q)
        force = force[force_idx]
        return jnp.where(is_valid, force, 0.0)

    return jax.vmap(force_one)(lambdas, qs_true, valid)


def reaction_force_loss(
    model,
    base,
    aux,
    idx_b,
    lambdas,
    qs_true,
    force_true,
    valid,
    force_components=(0, 1, 2),
    force_sign=1.0,
):
    pred_force = predict_reaction_force(
        model,
        base,
        aux,
        lambdas,
        qs_true,
        idx_b,
        valid,
        force_components=force_components,
        force_sign=force_sign,
    )

    if force_true.ndim == 1:
        pred_force = pred_force[:, 0]
    else:
        n_force = min(force_true.shape[-1], pred_force.shape[-1])
        force_true = force_true[:, :n_force]
        pred_force = pred_force[:, :n_force]

    err = (pred_force - force_true) ** 2
    if force_true.ndim == 1:
        masked_err = jnp.where(valid, err, 0.0)
    else:
        masked_err = jnp.where(valid[..., None], err, 0.0)
    return jnp.sum(masked_err) / jnp.maximum(jnp.sum(valid), 1.0)


def force_match_diagnostics(
    model,
    base,
    aux,
    data: Dataset,
    traj_idx=0,
    force_components=(0, 1, 2),
    force_sign=1.0,
):
    """
    Compare learned reaction forces against measured forces for one trajectory.

    The returned MSE values compare both sign conventions:
      mse_same_sign:     pred_force vs force_true
      mse_opposite_sign: -pred_force vs force_true
    If mse_opposite_sign is smaller, train/evaluate with force_sign=-force_sign.
    """
    if data.forces is None:
        raise ValueError("Dataset has no force data. Load with force_key='F' or include an F/force array.")

    if data.idx_b.ndim == 1:
        idx_b = data.idx_b
    else:
        idx_b = data.idx_b[traj_idx]

    if data.lambdas.ndim == 1:
        lambdas = data.lambdas
    else:
        lambdas = data.lambdas[traj_idx]

    qs_true = data.qs[traj_idx]
    force_true = data.forces[traj_idx]
    valid = data.valid[traj_idx]

    pred_force = predict_reaction_force(
        model,
        base,
        aux,
        lambdas,
        qs_true,
        idx_b,
        valid,
        force_components=force_components,
        force_sign=force_sign,
    )

    if force_true.ndim == 1:
        pred_cmp = pred_force[:, 0]
    else:
        n_force = min(force_true.shape[-1], pred_force.shape[-1])
        force_true = force_true[:, :n_force]
        pred_cmp = pred_force[:, :n_force]

    mask = valid.astype(pred_cmp.dtype)
    if force_true.ndim > 1:
        mask = mask[..., None]

    same_err = (pred_cmp - force_true) ** 2
    opposite_err = (-pred_cmp - force_true) ** 2
    denom = jnp.maximum(jnp.sum(mask), 1.0)
    mse_same = jnp.sum(jnp.where(mask.astype(bool), same_err, 0.0)) / denom
    mse_opposite = jnp.sum(jnp.where(mask.astype(bool), opposite_err, 0.0)) / denom
    best_sign = jnp.where(mse_same <= mse_opposite, force_sign, -force_sign)

    return {
        "lambdas": lambdas,
        "valid": valid,
        "force_true": force_true,
        "force_pred": pred_cmp,
        "mse_same_sign": mse_same,
        "mse_opposite_sign": mse_opposite,
        "rmse_same_sign": jnp.sqrt(mse_same),
        "rmse_opposite_sign": jnp.sqrt(mse_opposite),
        "best_force_sign": best_sign,
    }


def print_force_match_diagnostics(
    model,
    base,
    aux,
    data: Dataset,
    traj_idx=0,
    force_components=(0, 1, 2),
    force_sign=1.0,
):
    diag = force_match_diagnostics(
        model,
        base,
        aux,
        data,
        traj_idx=traj_idx,
        force_components=force_components,
        force_sign=force_sign,
    )
    mse_same = float(diag["mse_same_sign"])
    mse_opp = float(diag["mse_opposite_sign"])
    best = float(diag["best_force_sign"])
    print(f"Force MSE, same sign:     {mse_same:.6e}")
    print(f"Force MSE, opposite sign: {mse_opp:.6e}")
    print(f"Suggested force_sign:     {best:+.0f}")
    return diag


def plot_force_match(
    model,
    base,
    aux,
    data: Dataset,
    traj_idx=0,
    force_components=(0, 1, 2),
    force_sign=1.0,
    use_best_sign=True,
    component_labels=("Fx", "Fy", "Fz"),
    ax=None,
):
    """
    Plot predicted and ground-truth reaction-force components after training.
    """
    import matplotlib.pyplot as plt

    diag = force_match_diagnostics(
        model,
        base,
        aux,
        data,
        traj_idx=traj_idx,
        force_components=force_components,
        force_sign=force_sign,
    )

    lambdas = np.asarray(diag["lambdas"])
    valid = np.asarray(diag["valid"], dtype=bool)
    force_true = np.asarray(diag["force_true"])
    force_pred = np.asarray(diag["force_pred"])

    if use_best_sign and float(diag["best_force_sign"]) != float(force_sign):
        force_pred = -force_pred

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    if force_true.ndim == 1:
        ax.plot(lambdas[valid], force_true[valid], "k-", label="true")
        ax.plot(lambdas[valid], force_pred[valid], "--", label="pred")
    else:
        labels = component_labels[: force_true.shape[-1]]
        for i, label in enumerate(labels):
            ax.plot(lambdas[valid], force_true[valid, i], "-", label=f"{label} true")
            ax.plot(lambdas[valid], force_pred[valid, i], "--", label=f"{label} pred")

    ax.set_xlabel("lambda")
    ax.set_ylabel("reaction force")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return diag, ax


# =========================================================
# Loss (MSE over trajectories)
# =========================================================
def _move_valid_prefix(qs, xb, lambdas, valid):
    """Move a left-padded valid suffix to the front, repeating the last frame at the tail."""
    T = valid.shape[0]
    steps = jnp.arange(T)
    n_valid = jnp.sum(valid)
    first_valid = jnp.argmax(valid)
    src_idx = jnp.minimum(first_valid + steps, T - 1)
    compact_valid = steps < n_valid
    return qs[src_idx], xb[src_idx], lambdas[src_idx], compact_valid


def _move_valid_prefix_with_force(qs, xb, lambdas, force, valid):
    T = valid.shape[0]
    steps = jnp.arange(T)
    n_valid = jnp.sum(valid)
    first_valid = jnp.argmax(valid)
    src_idx = jnp.minimum(first_valid + steps, T - 1)
    compact_valid = steps < n_valid
    return qs[src_idx], xb[src_idx], lambdas[src_idx], force[src_idx], compact_valid


def traj_loss(
    model,
    base,
    aux,
    idx_b,
    xb,
    lambdas,
    qs_true,
    valid,
    force_true=None,
    max_dlambda=5e-3,
    iters=5,
    ls_steps=10,
    abs_tol=1e-8,
    rel_tol=1e-6,
    fail_on_nonconvergence=False,
    early_stop=False,
    hessian_reg_strength=0.0,
    hessian_reg_key=None,
    hessian_reg_probes=1,
    force_loss_strength=1.0,
    force_components=(0, 1, 2),
    force_sign=1.0,
):
    if force_true is None:
        qs_true, xb, lambdas, valid = _move_valid_prefix(qs_true, xb, lambdas, valid)
    else:
        qs_true, xb, lambdas, force_true, valid = _move_valid_prefix_with_force(
            qs_true,
            xb,
            lambdas,
            force_true,
            valid,
        )

    bc = djx.DirectBC(idx_b=idx_b, xb=xb, lambdas=lambdas)
    rod = base.with_bc(bc)

    qs_pred = rod.solve(
        model,
        lambdas,
        aux,
        max_dlambda=max_dlambda,
        iters=iters,
        ls_steps=ls_steps,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        fail_on_nonconvergence=fail_on_nonconvergence,
        early_stop=early_stop,
    )
    err = (qs_pred - qs_true) ** 2
    masked_err = jnp.where(valid[..., None], err, 0.0)
    mse = jnp.sum(masked_err) / jnp.sum(valid)

    if force_true is not None and force_loss_strength != 0.0:
        force_mse = reaction_force_loss(
            model,
            base,
            aux,
            idx_b,
            lambdas,
            qs_true,
            force_true,
            valid,
            force_components=force_components,
            force_sign=force_sign,
        )
        mse = mse + force_loss_strength * force_mse

    if hessian_reg_strength == 0.0:
        return mse

    hessian_reg = energy_hessian_spectral_regularizer(
        model,
        rod,
        aux,
        lambdas,
        qs_pred,
        valid,
        hessian_reg_key,
        n_probes=hessian_reg_probes,
    )
    return mse + hessian_reg_strength * hessian_reg


def dataset_loss(
    model,
    base,
    aux,
    data: Dataset,
    max_dlambda=5e-3,
    iters=5,
    ls_steps=10,
    abs_tol=1e-8,
    rel_tol=1e-6,
    fail_on_nonconvergence=False,
    early_stop=False,
    hessian_reg_strength=0.0,
    hessian_reg_key=None,
    hessian_reg_probes=1,
    force_loss_strength=1.0,
    force_components=(0, 1, 2),
    force_sign=1.0,
):
    n_traj = data.qs.shape[0]

    # handle shared vs per-trajectory idx_b
    if data.idx_b.ndim == 1:
        idx_all = jnp.broadcast_to(data.idx_b, (n_traj, data.idx_b.shape[0]))
    else:
        idx_all = data.idx_b

    # handle shared vs per-trajectory lambdas
    if data.lambdas.ndim == 1:
        lam_all = jnp.broadcast_to(data.lambdas, (n_traj, data.lambdas.shape[0]))
    else:
        lam_all = data.lambdas

    if hessian_reg_key is None:
        hessian_reg_key = jax.random.PRNGKey(0)
    reg_keys = jax.random.split(hessian_reg_key, n_traj)

    if data.forces is None:
        losses = jax.vmap(
            lambda ib, xb, qs, lam, valid, reg_key: traj_loss(
                model,
                base,
                aux,
                ib,
                xb,
                lam,
                qs,
                valid,
                max_dlambda=max_dlambda,
                iters=iters,
                ls_steps=ls_steps,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
                fail_on_nonconvergence=fail_on_nonconvergence,
                early_stop=early_stop,
                hessian_reg_strength=hessian_reg_strength,
                hessian_reg_key=reg_key,
                hessian_reg_probes=hessian_reg_probes,
                force_loss_strength=0.0,
                force_components=force_components,
                force_sign=force_sign,
            )
        )(idx_all, data.xb, data.qs, lam_all, data.valid, reg_keys)
    else:
        losses = jax.vmap(
            lambda ib, xb, qs, lam, valid, force, reg_key: traj_loss(
                model,
                base,
                aux,
                ib,
                xb,
                lam,
                qs,
                valid,
                force_true=force,
                max_dlambda=max_dlambda,
                iters=iters,
                ls_steps=ls_steps,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
                fail_on_nonconvergence=fail_on_nonconvergence,
                early_stop=early_stop,
                hessian_reg_strength=hessian_reg_strength,
                hessian_reg_key=reg_key,
                hessian_reg_probes=hessian_reg_probes,
                force_loss_strength=force_loss_strength,
                force_components=force_components,
                force_sign=force_sign,
            )
        )(idx_all, data.xb, data.qs, lam_all, data.valid, data.forces, reg_keys)

    return jnp.mean(losses)

# =========================================================
# Training
# =========================================================
def train_model(
    properties,
    model_cls,
    params: ModelParams,
    train_file,
    valid_file,
    force_key=None,
    n_epochs=100,
    lr=1e-2,
    snapshot_fn=None,
    snapshot_every=None,
    snapshot_epochs=None,
    snapshot_before_training=False,
    epoch_callback=None,
    valid_every=1,
    max_dlambda=5e-3,
    iters=5,
    ls_steps=10,
    abs_tol=1e-4,
    rel_tol=1e-4,
    fail_on_nonconvergence=False,
    early_stop=False,
    hessian_reg_strength=0.0,
    hessian_reg_probes=1,
    hessian_reg_seed=0,
    force_loss_strength=1.0,
    force_components=(0, 1, 2),
    force_sign=1.0,
):
    # --- setup ---
    base, aux = get_slinky(properties)
    train = Dataset.load(train_file, force_key=force_key)
    valid = Dataset.load(valid_file, force_key=force_key)
    validate_dataset_compatibility(base, train, "train")
    validate_dataset_compatibility(base, valid, "valid")

    model = model_cls(params)


    schedule = optax.cosine_decay_schedule(
        init_value=lr,
        decay_steps=n_epochs + 1,
        alpha=0.1,
    )
    # Optax: Adam
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule),
    ) 

    # # Optax: AdaBelief
    # opt = optax.chain(
    #     optax.clip_by_global_norm(1.0),
    #     optax.adabelief(learning_rate=lr),
    # )

    opt_state = opt.init(model)
    hessian_reg_key = jax.random.PRNGKey(hessian_reg_seed)

    # --- training step ---
    @eqx.filter_jit
    def step(model, opt_state, reg_key):
        loss, grads = eqx.filter_value_and_grad(
            lambda m: dataset_loss(
                m,
                base,
                aux,
                train,
                max_dlambda=max_dlambda,
                iters=iters,
                ls_steps=ls_steps,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
                fail_on_nonconvergence=fail_on_nonconvergence,
                early_stop=early_stop,
                hessian_reg_strength=hessian_reg_strength,
                hessian_reg_key=reg_key,
                hessian_reg_probes=hessian_reg_probes,
                force_loss_strength=force_loss_strength,
                force_components=force_components,
                force_sign=force_sign,
            )
        )(model)

        updates, opt_state = opt.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        return model, opt_state, loss

    train_hist = []
    valid_hist = []

    last_val_loss = jnp.nan
    snapshot_epoch_set = None if snapshot_epochs is None else set(snapshot_epochs)

    if snapshot_fn is not None and snapshot_before_training:
        snapshot_fn(
            model=model,
            epoch=-1,
            base=base,
            aux=aux,
            train=train,
            valid=valid,
            train_loss=jnp.nan,
            val_loss=jnp.nan,
        )

    if epoch_callback is not None:
        epoch_callback(
            model=model,
            epoch=-1,
            train_loss=jnp.nan,
            val_loss=jnp.nan,
        )

    for i in range(n_epochs):
        hessian_reg_key, step_reg_key = jax.random.split(hessian_reg_key)
        model, opt_state, train_loss = step(model, opt_state, step_reg_key)
        train_hist.append(train_loss)

        do_valid = (valid_every is not None) and (
            (i % valid_every == 0) or (i == n_epochs - 1)
        )

        if do_valid:
            last_val_loss = dataset_loss(
                model,
                base,
                aux,
                valid,
                max_dlambda=max_dlambda,
                iters=iters,
                ls_steps=ls_steps,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
                fail_on_nonconvergence=False,
                early_stop=early_stop,
                force_loss_strength=force_loss_strength,
                force_components=force_components,
                force_sign=force_sign,
            )

        valid_hist.append(last_val_loss)

        if i % 100 == 0:
            print(
                f"Epoch {i:03d} | Train: {float(train_loss):.3e} | "
                f"Valid: {float(last_val_loss):.3e}"
            )

        # optional snapshot hook
        should_snapshot = False
        if snapshot_fn is not None:
            if snapshot_every is not None and ((i % snapshot_every == 0) or (i == n_epochs - 1)):
                should_snapshot = True
            if snapshot_epoch_set is not None and i in snapshot_epoch_set:
                should_snapshot = True

        if should_snapshot:
            snapshot_fn(
                model=model,
                epoch=i,
                base=base,
                aux=aux,
                train=train,
                valid=valid,
                train_loss=train_loss,
                val_loss=last_val_loss,
            )

        if epoch_callback is not None:
            epoch_callback(
                model=model,
                epoch=i,
                train_loss=train_loss,
                val_loss=last_val_loss,
            )

    return model, train_hist, valid_hist

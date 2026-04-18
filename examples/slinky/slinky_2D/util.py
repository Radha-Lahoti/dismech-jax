import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import dismech_jax as djx

from Energy_NN_architectures import ModelParams


# =========================================================
# Dataset (ONLY direct BC)
# =========================================================
class Dataset(eqx.Module):
    qs: jax.Array        # (n_traj, T, dof)
    xb: jax.Array        # (n_traj, T, n_b)
    idx_b: jax.Array     # (n_b,) or (n_traj, n_b)
    lambdas: jax.Array   # (n_traj, T) or (T,)
    valid: jax.Array     # (n_traj, T)

    @staticmethod
    def load(path):
        data = np.load(path)
        return Dataset(
            qs=jnp.asarray(data["qs"]),
            xb=jnp.asarray(data["xb"]),
            idx_b=jnp.asarray(data["idx_b"]),
            lambdas=jnp.asarray(data["lambdas"]),
            valid=jnp.asarray(data["valid"]),
        )


def validate_dataset_compatibility(base, data: Dataset, label: str):
    expected_dof = base.q0.shape[0]
    actual_dof = data.qs.shape[-1]
    if actual_dof != expected_dof:
        inferred_n = (actual_dof + 1) // 4
        base_n = (expected_dof + 1) // 4
        raise ValueError(
            f"{label} dataset dof ({actual_dof}) does not match base rod dof ({expected_dof}). "
            f"This dataset appears to require N={inferred_n} nodes, but the current properties use N={base_n}."
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
        )

    pred = jax.vmap(predict_one)(idx_all, xb, lam_all)
    return pred

# =========================================================
# Loss (MSE over trajectories)
# =========================================================
def traj_loss(
    model,
    base,
    aux,
    idx_b,
    xb,
    lambdas,
    qs_true,
    valid,
    max_dlambda=5e-3,
    iters=5,
    ls_steps=10,
    abs_tol=1e-8,
    rel_tol=1e-6,
    fail_on_nonconvergence=False,
):
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
    )
    err = (qs_pred - qs_true) ** 2
    masked_err = jnp.where(valid[..., None], err, 0.0)
    return jnp.sum(masked_err) / jnp.sum(valid)


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

    losses = jax.vmap(
        lambda ib, xb, qs, lam, valid: traj_loss(
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
        )
    )(idx_all, data.xb, data.qs, lam_all, data.valid)

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
    n_epochs=100,
    lr=1e-2,
    snapshot_fn=None,
    snapshot_every=None,
    snapshot_epochs=None,
    snapshot_before_training=False,
    valid_every=1,
    max_dlambda=5e-3,
    iters=5,
    ls_steps=10,
    abs_tol=1e-8,
    rel_tol=1e-6,
    fail_on_nonconvergence=False,
):
    # --- setup ---
    base, aux = get_slinky(properties)
    train = Dataset.load(train_file)
    valid = Dataset.load(valid_file)
    validate_dataset_compatibility(base, train, "train")
    validate_dataset_compatibility(base, valid, "valid")

    model = model_cls(params)

    # Optax: AdaBelief
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adabelief(learning_rate=lr),
    )

    opt_state = opt.init(model)

    # --- training step ---
    @eqx.filter_jit
    def step(model, opt_state):
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

    for i in range(n_epochs):
        model, opt_state, train_loss = step(model, opt_state)
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
                fail_on_nonconvergence=fail_on_nonconvergence,
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

    return model, train_hist, valid_hist

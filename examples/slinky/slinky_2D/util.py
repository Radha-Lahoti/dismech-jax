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
    lambdas: jax.Array   # (T,)
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
        f = jnp.array([
            0, 0, mass / 4 * -9.81,
            0, 0, 0,
            mass / 2 * -9.81,
            0, 0, 0,
            mass / 4 * -9.81,
        ])
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
):
    bc = djx.BatchedDirectBC(idx_b=idx_b, xb=xb, lambdas=lambdas)
    rod = base.with_bc(bc)
    pred = rod.solve(
        model,
        lambdas,
        aux,
        max_dlambda=max_dlambda,
        iters=iters,
        ls_steps=ls_steps,
    )
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
):
    n_traj = data.qs.shape[0]

    # handle shared vs per-trajectory idx_b
    if data.idx_b.ndim == 1:
        idx_all = jnp.broadcast_to(data.idx_b, (n_traj, data.idx_b.shape[0]))
    else:
        idx_all = data.idx_b

    losses = jax.vmap(
        lambda ib, xb, qs, _lambda, valid: traj_loss(
            model,
            base,
            aux,
            ib,
            xb,
            _lambda,
            qs,
            valid,
            max_dlambda=max_dlambda,
            iters=iters,
            ls_steps=ls_steps,
        )
    )(idx_all, data.xb, data.qs, data.lambdas, data.valid)

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
    valid_every=1,
    max_dlambda=5e-3,
    iters=5,
    ls_steps=10,
):
    # --- setup ---
    base, aux = get_slinky(properties)
    train = Dataset.load(train_file)
    valid = Dataset.load(valid_file)

    model = model_cls(params)

    schedule = optax.cosine_decay_schedule(
        init_value=lr,
        decay_steps=n_epochs + 1,
        alpha=0.1,
    )

    opt = optax.adam(schedule)
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
            )
        )(model)

        updates, opt_state = opt.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        return model, opt_state, loss

    train_hist = []
    valid_hist = []

    last_val_loss = jnp.nan

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
            )

        valid_hist.append(last_val_loss)

        if i % 10 == 0:
            print(
                f"Epoch {i:03d} | Train: {float(train_loss):.3e} | "
                f"Valid: {float(last_val_loss):.3e}"
            )

        # optional snapshot hook
        if snapshot_fn is not None and snapshot_every is not None:
            if (i % snapshot_every == 0) or (i == n_epochs - 1):
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
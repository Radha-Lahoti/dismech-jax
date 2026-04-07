"""
Unified training utilities for slinky / ribbon triplet models.

Supports:

* **Data layout** (auto-detected from NPZ): single trajectory with
  :class:`dismech_jax.BatchedLinearBC`, multiset linear BC (``xb_c`` / ``xb_m``),
  or multiset **direct** BC (``xb`` time series).
* **Rod geometry** via ``rod_preset`` (default slinky, ribbon experiment, ribbon strip).
* **Optional spectral stiffness regularization** (``alpha_spec > 0``) for models that
  expose ``get_K_matrix(del_strain)``.

Legacy modules ``util_multiset.py``, ``util_reg_loss.py``, ``util_multiset_copy.py``,
``util_copy.py``, and ``util_multiset_ribbon.py`` re-export this API for old notebooks.
"""

from __future__ import annotations

import inspect
import warnings
from abc import abstractmethod
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from beartype import beartype
from jaxtyping import Float, TypeCheckError, jaxtyped

import dismech_jax

RodPreset = Literal["slinky", "ribbon_experiment", "ribbon_strip"]
DatasetLayout = Literal["single_linear", "multi_linear", "multi_direct"]


# --- Triplet model base ---------------------------------------------------


class TripletModel(eqx.Module):
    """NN strain-energy density base class (scalar energy from 5-vector strain)."""

    def __init__(self, der_K: jax.Array, key: jax.Array): ...

    @abstractmethod
    def __call__(self, del_strain: Float[jax.Array, "5"]) -> Float[jax.Array, ""]: ...

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        parent_annotations = TripletModel.__call__.__annotations__
        if "__call__" in cls.__dict__:
            child_call = cls.__dict__["__call__"]
            child_call.__annotations__ = parent_annotations.copy()
            cls.__call__ = jaxtyped(typechecker=beartype)(child_call)


# --- Test case container --------------------------------------------------


class TestCase(eqx.Module):
    """
    Training/validation pack loaded from ``.npz``.

    Exactly one BC style is active; see :meth:`from_npz` for detection rules.
    """

    layout: str
    qs: jax.Array
    lambdas: jax.Array | None
    bc: dismech_jax.BatchedLinearBC | None = None
    idx_b: jax.Array | None = None
    xb_c: jax.Array | None = None
    xb_m: jax.Array | None = None
    xb: jax.Array | None = None

    @classmethod
    def from_npz(cls, filename: str) -> TestCase:
        data = np.load(filename)
        qs = jnp.asarray(data["qs"])
        lambdas = jnp.asarray(data["lambdas"]) if "lambdas" in data else None

        if qs.ndim == 2:
            idx_b = jnp.asarray(data["idx_b"])
            xb_c = jnp.asarray(data["xb_c"])
            xb_m = jnp.asarray(data["xb_m"])
            bc = dismech_jax.BatchedLinearBC(idx_b=idx_b, xb_c=xb_c, xb_m=xb_m)
            return cls(
                layout="single_linear",
                qs=qs,
                lambdas=lambdas,
                bc=bc,
            )

        if qs.ndim != 3:
            raise ValueError(f"{filename}: qs must be 2D or 3D, got {qs.shape}")

        idx_b = jnp.asarray(data["idx_b"])

        if "xb" in data:
            xb = jnp.asarray(data["xb"])
            if xb.ndim != 3:
                raise ValueError(f"{filename}: expected xb (n_traj, n_lambda, n_b), got {xb.shape}")
            _check_idx_xb(qs, idx_b, xb, filename)
            if lambdas is not None and lambdas.shape[0] != qs.shape[1]:
                raise ValueError("lambdas length must match qs.shape[1]")
            return cls(
                layout="multi_direct",
                qs=qs,
                lambdas=lambdas,
                idx_b=idx_b,
                xb=xb,
            )

        if "xb_c" in data and "xb_m" in data:
            xb_c = jnp.asarray(data["xb_c"])
            xb_m = jnp.asarray(data["xb_m"])
            if xb_c.ndim != 2:
                print("xb_c: ", xb_c)
                xb_c = jnp.broadcast_to(xb_c, (qs.shape[0], xb_c.shape[0]))
                warnings.warn(
                    f"{filename}: broadcast xb_c to (n_traj, n_b) from {xb_c.shape}"
                )
            if xb_m.ndim != 3:
                print("xb_m: ", xb_m)
                xb_m = jnp.broadcast_to(xb_m, (qs.shape[0], xb_m.shape[0]))
                warnings.warn(
                    f"{filename}: broadcast xb_m to (n_traj, n_b) from {xb_m.shape}"
                )
            if xb_m.shape != xb_c.shape:
                raise ValueError(
                    f"{filename}: xb_c.shape={xb_c.shape} must match xb_m.shape={xb_m.shape}"
                )
            _check_idx_linear(qs, idx_b, xb_c, filename)
            if lambdas is not None and lambdas.shape[0] != qs.shape[1]:
                raise ValueError("lambdas length must match qs.shape[1]")
            return cls(
                layout="multi_linear",
                qs=qs,
                lambdas=lambdas,
                idx_b=idx_b,
                xb_c=xb_c,
                xb_m=xb_m,
            )

        raise ValueError(
            f"{filename}: multiset NPZ must contain 'xb' (direct) or 'xb_c'/'xb_m' (linear)"
        )


def _check_idx_xb(qs: jax.Array, idx_b: jax.Array, xb: jax.Array, filename: str) -> None:
    n_traj, n_lam, _ = qs.shape
    if xb.shape[0] != n_traj or xb.shape[1] != n_lam:
        raise ValueError(f"{filename}: xb shape {xb.shape} incompatible with qs {qs.shape}")
    if idx_b.ndim == 1:
        if idx_b.shape[0] != xb.shape[2]:
            raise ValueError(f"{filename}: idx_b vs xb mismatch")
    elif idx_b.ndim == 2:
        if idx_b.shape != (n_traj, xb.shape[2]):
            raise ValueError(f"{filename}: idx_b shape {idx_b.shape}")
    else:
        raise ValueError(f"{filename}: bad idx_b ndim {idx_b.ndim}")


def _check_idx_linear(qs: jax.Array, idx_b: jax.Array, xb_c: jax.Array, filename: str) -> None:
    n_traj, _, _ = qs.shape
    if xb_c.shape[0] != n_traj:
        raise ValueError(f"{filename}: xb_c.shape[0] != n_traj")
    if idx_b.ndim == 1:
        if idx_b.shape[0] != xb_c.shape[1]:
            raise ValueError(f"{filename}: idx_b vs xb_c mismatch")
    elif idx_b.ndim == 2:
        if idx_b.shape != xb_c.shape:
            raise ValueError(f"{filename}: idx_b vs xb_c shape")
    else:
        raise ValueError(f"{filename}: bad idx_b ndim {idx_b.ndim}")


# --- Rod presets ----------------------------------------------------------


def get_base_rod(preset: RodPreset = "slinky") -> tuple[dismech_jax.Rod, jax.Array, Any]:
    """Return ``(base_rod, aux, der)`` for the chosen experiment template."""
    if preset == "slinky":
        geom = dismech_jax.Geometry(0.5, 5e-3)
        mat = dismech_jax.Material(1273.52, 1e7)
        temp, aux = dismech_jax.Rod.from_geometry(geom, mat, N=3)
        mass = 0.647
        f_new = jnp.array(
            [
                0.0,
                0.0,
                mass / 4 * -9.81,
                0.0,
                0.0,
                0.0,
                mass / 2 * -9.81,
                0.0,
                0.0,
                0.0,
                mass / 4 * -9.81,
            ]
        )
        base = eqx.tree_at(lambda r: r.E_ext, temp, dismech_jax.Gravity(f_new))
        der = base.get_DER(geom, mat)
        return base, aux, der

    if preset == "ribbon_experiment":
        geom = dismech_jax.Geometry(0.225, 5e-3)
        mat = dismech_jax.Material(1273.52, 1e7)
        temp, aux = dismech_jax.Rod.from_endpoints(
            start=jnp.array([0.0, 0.0, 0.0]),
            end=jnp.array([0.3024303, 0.0, 0.02364909]),
            material=mat,
            N=3,
        )
        mass = 0.032
        f_new = jnp.array(
            [
                0.0,
                0.0,
                mass / 4 * -9.81,
                0.0,
                0.0,
                0.0,
                mass / 2 * -9.81,
                0.0,
                0.0,
                0.0,
                mass / 4 * -9.81,
            ]
        )
        base = eqx.tree_at(lambda r: r.E_ext, temp, dismech_jax.Gravity(f_new))
        der = base.get_DER(geom, mat)
        return base, aux, der

    if preset == "ribbon_strip":
        b = 0.02
        h = 0.001
        geom = dismech_jax.Geometry(
            0.1,
            axs=b * h,
            ixs1=b * h**3 / 12,
            ixs2=h * b**3 / 12,
            jxs=b * h**3 / 6,
        )
        mat = dismech_jax.Material(800, 2e7, 0.5)
        base, aux = dismech_jax.Rod.from_geometry(geom, mat, N=21)
        der = base.get_DER(geom, mat)
        return base, aux, der

    raise ValueError(f"Unknown rod_preset: {preset!r}")


# --- Model construction / validation --------------------------------------


def create_model(
    cls: type,
    init_K: jax.Array,
    key: jax.Array,
    base: dismech_jax.Rod,
    model_init_kwargs: dict[str, Any] | None = None,
) -> eqx.Module:
    """Instantiate ``cls``; prefer ``cls(der_K, key, **model_init_kwargs)`` (matches old notebooks).

    If that raises ``TypeError``, retry with ``l_k`` from ``base.triplets.l_k`` for
    models that require an explicit length scale.
    """
    model_init_kwargs = dict(model_init_kwargs or {})
    lk = float(base.triplets.l_k[0, 0])
    attempts = [
        lambda: cls(der_K=init_K, key=key, **model_init_kwargs),
        lambda: cls(der_K=init_K, key=key, l_k=lk, **model_init_kwargs),
    ]
    last_err: Exception | None = None
    for fn in attempts:
        try:
            return fn()
        except TypeError as e:
            last_err = e
            continue
    raise TypeError(f"Could not construct {cls}: {last_err}")


def validate_model(
    cls: type,
    der_K: jax.Array,
    *,
    rod_preset: RodPreset = "slinky",
    model_init_kwargs: dict[str, Any] | None = None,
) -> None:
    if not issubclass(cls, TripletModel):
        raise ValueError(f"{cls} is not a TripletModel subclass")
    base, _, _ = get_base_rod(rod_preset)
    obj = create_model(cls, jnp.asarray(der_K), jax.random.PRNGKey(42), base, model_init_kwargs)
    try:
        out = obj(jnp.zeros(5))
    except TypeCheckError as e:
        raise ValueError(f"__call__ type error: {e}") from e
    if jnp.shape(out) != ():
        raise ValueError(f"__call__ must return scalar, got {jnp.shape(out)}")


def stiffness_trace(model: eqx.Module) -> jax.Array | None:
    """Best-effort stiffness vector for logging."""
    if hasattr(model, "get_K_entries"):
        fn = model.get_K_entries
        sig = inspect.signature(fn)
        if len(sig.parameters) == 0:
            return fn()
        return fn(jnp.zeros(5))
    if hasattr(model, "K0"):
        return getattr(model, "K0")
    if hasattr(model, "K"):
        return getattr(model, "K")
    return None


# --- Loss pieces ----------------------------------------------------------


def _lambdas_for(dataset: TestCase) -> jax.Array:
    if dataset.lambdas is not None:
        return dataset.lambdas
    return jnp.linspace(0.0, 1.0, dataset.qs.shape[0 if dataset.layout == "single_linear" else 1])


def _idx_b_all(dataset: TestCase) -> jax.Array:
    assert dataset.idx_b is not None
    n_traj = dataset.qs.shape[0]
    if dataset.idx_b.ndim == 1:
        return jnp.broadcast_to(dataset.idx_b[None, :], (n_traj, dataset.idx_b.shape[0]))
    return dataset.idx_b


def _loss_single(model, base, aux, dataset: TestCase) -> jax.Array:
    assert dataset.bc is not None
    rods = base.with_bc(dataset.bc)
    lam = _lambdas_for(dataset)
    pred = rods.solve(model, lam, aux, max_dlambda=5e-3, iters=5, ls_steps=10)
    return jnp.mean(jnp.square(pred - dataset.qs))


def _traj_loss_linear(model, base, aux, idx_b, xb_c, xb_m, lam, truth_qs) -> jax.Array:
    bc = dismech_jax.BatchedLinearBC(idx_b=idx_b, xb_c=xb_c, xb_m=xb_m)
    rod = base.with_bc(bc)
    pred = rod.solve(model, lam, aux, max_dlambda=5e-3, iters=5, ls_steps=10)
    return jnp.mean(jnp.square(pred - truth_qs))


def _dataset_loss_linear(model, base, aux, dataset: TestCase) -> jax.Array:
    assert dataset.xb_c is not None and dataset.xb_m is not None
    lam = _lambdas_for(dataset)
    idx_all = _idx_b_all(dataset)
    losses = jax.vmap(
        lambda ib, xc, xm, tq: _traj_loss_linear(model, base, aux, ib, xc, xm, lam, tq)
    )(idx_all, dataset.xb_c, dataset.xb_m, dataset.qs)
    return jnp.mean(losses)


def _traj_loss_direct(model, base, aux, idx_b, xb, lam, truth_qs) -> jax.Array:
    bc = dismech_jax.DirectBC(idx_b=idx_b, xb=xb, lambdas=lam)
    rod = base.with_bc(bc)
    pred = rod.solve(model, lam, aux, max_dlambda=5e-3, iters=5, ls_steps=10)
    return jnp.mean(jnp.square(pred - truth_qs))


def _dataset_loss_direct(model, base, aux, dataset: TestCase) -> jax.Array:
    assert dataset.xb is not None
    lam = _lambdas_for(dataset)
    idx_all = _idx_b_all(dataset)
    losses = jax.vmap(
        lambda ib, xb_i, tq: _traj_loss_direct(model, base, aux, ib, xb_i, lam, tq)
    )(idx_all, dataset.xb, dataset.qs)
    return jnp.mean(losses)


def dataset_mse_loss(model, base, aux, dataset: TestCase) -> jax.Array:
    if dataset.layout == "single_linear":
        return _loss_single(model, base, aux, dataset)
    if dataset.layout == "multi_linear":
        return _dataset_loss_linear(model, base, aux, dataset)
    if dataset.layout == "multi_direct":
        return _dataset_loss_direct(model, base, aux, dataset)
    raise ValueError(dataset.layout)


def spectral_spread_penalty(
    k_mat: jax.Array,
    eps: float = 1e-8,
    max_log_spread: float | None = 2.0,
) -> jax.Array:
    k_mat = 0.5 * (k_mat + k_mat.T)
    eigs = jnp.linalg.eigvalsh(k_mat)
    lam_min = jnp.maximum(eigs[0], eps)
    lam_max = jnp.maximum(eigs[-1], eps)
    log_spread = jnp.log(lam_max) - jnp.log(lam_min)
    if max_log_spread is None:
        return log_spread**2
    excess = jnp.maximum(log_spread - max_log_spread, 0.0)
    return excess**2


def _build_aux_history(aux0, qs: jax.Array):
    def step(aux_prev, q):
        aux_next = jax.vmap(lambda a: a.update(q))(aux_prev)
        return aux_next, aux_next

    _, aux_hist = jax.lax.scan(step, aux0, qs)
    return aux_hist


def precompute_strain_bank(base: dismech_jax.Rod, aux0, dataset: TestCase) -> jax.Array:
    """Flattened ``del_strain`` rows for optional spectral regularization."""
    if dataset.layout == "single_linear":
        qs_b = dataset.qs[None, ...]
    else:
        qs_b = dataset.qs

    def one_traj(qs_traj):
        aux_hist = _build_aux_history(aux0, qs_traj)
        dhist = base.get_del_strain_history(qs_traj, aux_hist)
        return dhist.reshape(-1, dhist.shape[-1])

    blocks = jax.vmap(one_traj)(qs_b)
    return blocks.reshape(-1, blocks.shape[-1])


def spectral_loss_from_bank(
    model: eqx.Module,
    bank: jax.Array,
    *,
    max_log_spread: float | None = 2.0,
) -> jax.Array:
    if not hasattr(model, "get_K_matrix"):
        raise TypeError("Spectral loss requires model.get_K_matrix(del_strain)")
    pens = jax.vmap(
        lambda ds: spectral_spread_penalty(model.get_K_matrix(ds), max_log_spread=max_log_spread)
    )(bank)
    return jnp.mean(pens)


def _alpha_spec_schedule(
    step_idx: jax.Array,
    alpha_spec_max: float,
    start_frac: float,
    ramp_frac: float,
    num_steps: int,
) -> jax.Array:
    start_step = start_frac * num_steps
    ramp_steps = jnp.maximum(ramp_frac * num_steps, 1.0)
    progress = jnp.clip((step_idx - start_step) / ramp_steps, 0.0, 1.0)
    return alpha_spec_max * progress


def _loss_with_spec(
    model,
    base,
    aux,
    dataset: TestCase,
    bank: jax.Array,
    alpha_spec: float,
    *,
    max_log_spread: float | None,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    data = dataset_mse_loss(model, base, aux, dataset)
    spec = spectral_loss_from_bank(model, bank, max_log_spread=max_log_spread)
    return data + alpha_spec * spec, (data, spec)


# --- Main training entry ---------------------------------------------------


def train_model(
    cls: type,
    key: jax.Array = jax.random.PRNGKey(42),
    train_file: str = "train.npz",
    valid_file: str = "valid.npz",
    n_epochs: int = 100,
    lr: float = 1e-2,
    init_K: jax.Array | None = None,
    *,
    rod_preset: RodPreset = "slinky",
    alpha_spec: float = 0.0,
    spec_start_frac: float = 0.2,
    spec_ramp_frac: float = 0.3,
    max_log_spread: float | None = 2.0,
    print_initial_debug: bool = False,
    validate_before_train: bool = True,
    model_init_kwargs: dict[str, Any] | None = None,
    val_interval: int = 10,
) -> tuple:
    """
    Train ``cls`` on NPZ data.

    Set ``validate_before_train=False`` to skip the pre-flight ``validate_model`` call
    (slightly faster startup once model construction is trusted).

    When ``alpha_spec == 0`` (default), returns
    ``(model, init_K_trace, train_hist, valid_hist)``.

    When ``alpha_spec > 0``, returns eight values: ``model``, initial stiffness
    trace, then train total/data/spec and valid total/data/spec histories
    (same layout as the old ``util_reg_loss`` helpers).
    """
    base, aux, der = get_base_rod(rod_preset)
    train = TestCase.from_npz(train_file)
    valid = TestCase.from_npz(valid_file)

    if init_K is None:
        if rod_preset == "ribbon_strip":
            ks = der.K
            init_K = jnp.array([ks[0], 0.0001, ks[2]])
        else:
            init_K = jnp.array([2.0, 0.01, 0.02])
    else:
        init_K = jnp.asarray(init_K)

    if validate_before_train:
        validate_model(cls, init_K, rod_preset=rod_preset, model_init_kwargs=model_init_kwargs)

    model = create_model(cls, init_K, key, base, model_init_kwargs)
    init_trace = stiffness_trace(model)
    if init_trace is None:
        init_trace = init_K

    if alpha_spec > 0 and not hasattr(model, "get_K_matrix"):
        raise TypeError("alpha_spec > 0 requires a model with get_K_matrix")

    schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=n_epochs + 1, alpha=0.1)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(model)

    if print_initial_debug:
        il = dataset_mse_loss(model, base, aux, train)
        print(f"Initial training loss: {float(il):.5e}")
        g = eqx.filter_grad(lambda m: dataset_mse_loss(m, base, aux, train))(model)
        print("grad tree:", jax.tree_util.tree_map(lambda x: jnp.all(jnp.isfinite(x)) if eqx.is_array(x) else None, g))

    if alpha_spec <= 0.0:

        @eqx.filter_jit
        def run_training(m, s, num_steps: int):
            def step(carry, step_idx):
                mm, ss = carry
                tr, gr = eqx.filter_value_and_grad(lambda md: dataset_mse_loss(md, base, aux, train))(mm)
                up, nss = optimizer.update(gr, ss, mm)
                nm = eqx.apply_updates(mm, up)
                is_v = (step_idx % val_interval) == 0
                vl = jax.lax.cond(
                    is_v,
                    lambda: dataset_mse_loss(nm, base, aux, valid),
                    lambda: -1.0,
                )
                lr_now = schedule(step_idx)

                def cb(si, lri, trn, va, kk):
                    print(
                        f"Step {int(si):<4} | LR: {float(lri):<10.3e} | "
                        f"Train: {float(trn):<12.5e} | Valid: {float(va):<12.5e} | K: {kk}"
                    )

                def log_step():
                    ksnap = stiffness_trace(nm)
                    if ksnap is None:
                        ksnap = jnp.array([jnp.nan])
                    return jax.debug.callback(cb, step_idx, lr_now, tr, vl, ksnap)

                jax.lax.cond(is_v, log_step, lambda: None)
                return (nm, nss), (tr, vl)

            (fm, fs), hist = jax.lax.scan(step, (m, s), jnp.arange(num_steps))
            return fm, fs, hist

        model, opt_state, (th, vh) = run_training(model, opt_state, n_epochs + 1)
        return model, init_trace, th, vh

    train_bank = precompute_strain_bank(base, aux, train)
    valid_bank = precompute_strain_bank(base, aux, valid)

    @eqx.filter_jit
    def run_training_spec(m, s, num_steps: int):
        def step(carry, step_idx):
            mm, ss = carry
            a_now = _alpha_spec_schedule(
                step_idx, alpha_spec, spec_start_frac, spec_ramp_frac, num_steps
            )
            (tot, (td, sp)), gr = eqx.filter_value_and_grad(
                lambda md: _loss_with_spec(
                    md, base, aux, train, train_bank, a_now, max_log_spread=max_log_spread
                ),
                has_aux=True,
            )(mm)
            up, nss = optimizer.update(gr, ss, mm)
            nm = eqx.apply_updates(mm, up)
            is_v = (step_idx % val_interval) == 0

            def val_fn():
                o = _loss_with_spec(
                    nm, base, aux, valid, valid_bank, a_now, max_log_spread=max_log_spread
                )
                return o[0], o[1][0], o[1][1]

            v_tot, v_d, v_s = jax.lax.cond(is_v, val_fn, lambda: (-1.0, -1.0, -1.0))
            lr_now = schedule(step_idx)

            def cb(si, lri, an, tt, td_, ts_, vt, vd_, vs_, kk):
                print(
                    f"Step {int(si):<4} | LR: {float(lri):<10.3e} | alpha_spec: {float(an):<10.3e} | "
                    f"TrainTot: {float(tt):<12.5e} | TrainData: {float(td_):<12.5e} | TrainSpec: {float(ts_):<12.5e} | "
                    f"ValidTot: {float(vt):<12.5e} | ValidData: {float(vd_):<12.5e} | "
                    f"ValidSpec: {float(vs_):<12.5e} | K: {kk}"
                )

            def log_step():
                ksnap = stiffness_trace(nm)
                if ksnap is None:
                    ksnap = jnp.array([jnp.nan])
                return jax.debug.callback(
                    cb,
                    step_idx,
                    lr_now,
                    a_now,
                    tot,
                    td,
                    sp,
                    v_tot,
                    v_d,
                    v_s,
                    ksnap,
                )

            jax.lax.cond(is_v, log_step, lambda: None)
            return (nm, nss), (tot, td, sp, v_tot, v_d, v_s)

        (fm, fs), hist = jax.lax.scan(step, (m, s), jnp.arange(num_steps))
        return fm, fs, hist

    model, opt_state, hist = run_training_spec(model, opt_state, n_epochs + 1)
    tt, td, ts, vt, vd, vs = hist
    return model, init_trace, tt, td, ts, vt, vd, vs


def debug_dataset_loss(model: eqx.Module, base: dismech_jax.Rod, aux: jax.Array, dataset: TestCase) -> jax.Array:
    """Verbose per-trajectory breakdown (direct multiset)."""
    if dataset.layout != "multi_direct":
        raise ValueError("debug_dataset_loss only supports multi_direct layout")
    lam = _lambdas_for(dataset)
    idx_all = _idx_b_all(dataset)
    losses = []
    for i in range(dataset.qs.shape[0]):
        idx_b_i = idx_all[i]
        xb_i = dataset.xb[i]
        truth_i = dataset.qs[i]
        bc = dismech_jax.DirectBC(idx_b=idx_b_i, xb=xb_i, lambdas=lam)
        rod = base.with_bc(bc)
        pred_i = rod.solve(model, lam, aux, max_dlambda=5e-3, iters=5, ls_steps=10)
        losses.append(jnp.mean(jnp.square(pred_i - truth_i)))
    return jnp.mean(jnp.array(losses))

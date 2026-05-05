import jax
import jax.numpy as jnp
import equinox as eqx


# =========================================================
# Energy as NN
# Enforce E(0)=0 by zero-referencing the learned correction:
#   E_corr(eps) = raw_net(eps) - raw_net(0)
# =========================================================
class MLP_Energy(eqx.Module):
    ws: tuple[jax.Array, ...]
    bs: tuple[jax.Array, ...]

    k0_raw: jax.Array
    which_case: str = eqx.field(static=True)

    def __init__(
        self,
        key,
        K_initial=1.0,
        which_case=None,
        hidden_sizes=(10, 10),
        weight_scale=0.01,
    ):
        if len(hidden_sizes) < 1:
            raise ValueError("hidden_sizes must contain at least one hidden layer.")

        hidden_sizes = tuple(hidden_sizes)
        layer_sizes = (1,) + hidden_sizes + (1,)
        keys = jax.random.split(key, len(layer_sizes) - 1)

        ws = []
        bs = []
        for k, (din, dout) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
            ws.append(weight_scale * jax.random.normal(k, (dout, din)))
            bs.append(jnp.zeros((dout,)))

        self.ws = tuple(ws)
        self.bs = tuple(bs)

        self.k0_raw = jnp.array(K_initial)
        self.which_case = which_case

    def _raw_mlp_energy(self, eps):
        x = jnp.atleast_1d(eps).reshape(1)

        z = x
        for w, b in zip(self.ws[:-1], self.bs[:-1]):
            z = jax.nn.softplus(w @ z + b)

        out = self.ws[-1] @ z + self.bs[-1]
        return jax.nn.softplus(out).squeeze()

    def mlp_energy(self, eps):
        # Zero-reference the learned energy correction so that:
        # mlp_energy(0) = 0
        return self._raw_mlp_energy(eps) - self._raw_mlp_energy(jnp.array(0.0))

    def baseline_energy(self, eps):
        eps_scalar = jnp.squeeze(eps)
        k0 = jax.nn.softplus(self.k0_raw)
        return 0.5 * k0 * eps_scalar**2

    def __call__(self, eps):
        if self.which_case == "baseline":
            return self.baseline_energy(eps)
        elif self.which_case == "mlp":
            return self.mlp_energy(eps)
        else:
            return self.baseline_energy(eps) + self.mlp_energy(eps)


class ICNN_Energy(eqx.Module):
    wxs: tuple[jax.Array, ...]
    wzs_raw: tuple[jax.Array, ...]
    bs: tuple[jax.Array, ...]

    k0_raw: jax.Array
    which_case: str = eqx.field(static=True)

    def __init__(
        self,
        key,
        K_initial=1.0,
        which_case=None,
        hidden_sizes=(10, 10),
        weight_scale=0.01,
    ):
        if len(hidden_sizes) < 1:
            raise ValueError("hidden_sizes must contain at least one hidden layer.")

        hidden_sizes = tuple(hidden_sizes)

        n_wx = len(hidden_sizes) + 1
        n_b = len(hidden_sizes) + 1
        n_wz = len(hidden_sizes)
        keys = jax.random.split(key, n_wx + n_b + n_wz)

        wx_keys = keys[:n_wx]
        wz_keys = keys[n_wx:n_wx + n_wz]

        wxs = []
        wzs_raw = []
        bs = []

        in_dim = 1

        first_hidden = hidden_sizes[0]
        wxs.append(weight_scale * jax.random.normal(wx_keys[0], (first_hidden, in_dim)))
        bs.append(jnp.zeros((first_hidden,)))

        for i in range(1, len(hidden_sizes)):
            prev_h = hidden_sizes[i - 1]
            curr_h = hidden_sizes[i]

            wzs_raw.append(
                weight_scale * jax.random.normal(wz_keys[i - 1], (curr_h, prev_h))
            )
            wxs.append(
                weight_scale * jax.random.normal(wx_keys[i], (curr_h, in_dim))
            )
            bs.append(jnp.zeros((curr_h,)))

        last_hidden = hidden_sizes[-1]
        wzs_raw.append(
            weight_scale * jax.random.normal(wz_keys[-1], (1, last_hidden))
        )
        wxs.append(weight_scale * jax.random.normal(wx_keys[-1], (1, in_dim)))
        bs.append(jnp.zeros((1,)))

        self.wxs = tuple(wxs)
        self.wzs_raw = tuple(wzs_raw)
        self.bs = tuple(bs)

        self.k0_raw = jnp.array(K_initial)
        self.which_case = which_case

    def _raw_icnn_energy(self, eps):
        x = jnp.atleast_1d(eps).reshape(1)

        z = jax.nn.softplus(self.wxs[0] @ x + self.bs[0])

        for i in range(1, len(self.bs) - 1):
            wz = jax.nn.softplus(self.wzs_raw[i - 1])
            z = jax.nn.softplus(wz @ z + self.wxs[i] @ x + self.bs[i])

        wz_out = jax.nn.softplus(self.wzs_raw[-1])
        out = wz_out @ z + self.wxs[-1] @ x + self.bs[-1]

        return jax.nn.softplus(out).squeeze()

    def icnn_energy(self, eps):
        # Subtract value AND gradient at 0 so that icnn_energy(0) = 0,
        # icnn_energy'(0) = 0. Combined with the convexity of the raw ICNN,
        # this makes 0 the global minimum, so icnn_energy(eps) >= 0 everywhere.
        eps0 = jnp.array(0.0)
        raw0 = self._raw_icnn_energy(eps0)
        grad0 = jax.grad(self._raw_icnn_energy)(eps0)
        eps_scalar = jnp.squeeze(eps)
        return self._raw_icnn_energy(eps) - raw0 - grad0 * eps_scalar

    def baseline_energy(self, eps):
        eps_scalar = jnp.squeeze(eps)
        k0 = jax.nn.softplus(self.k0_raw)
        return 0.5 * k0 * eps_scalar**2

    def __call__(self, eps):
        if self.which_case == "baseline":
            return self.baseline_energy(eps)
        elif self.which_case == "icnn":
            return self.icnn_energy(eps)
        else:
            return self.baseline_energy(eps) + self.icnn_energy(eps)


# =========================================================
# Stiffness as NN
# No zero-offset fix needed here because E(eps)=K(eps)*eps^2
# already guarantees E(0)=0.
# =========================================================
class MLP_Stiffness(eqx.Module):
    ws: tuple[jax.Array, ...]
    bs: tuple[jax.Array, ...]

    k0_raw: jax.Array
    which_case: str = eqx.field(static=True)
    alpha_corr: jax.Array

    def __init__(
        self,
        key,
        K_initial=1.0,
        which_case=None,
        hidden_sizes=(10,),
        weight_scale=0.01,
        alpha_corr = 0.01
    ):
        if len(hidden_sizes) < 1:
            raise ValueError("Need at least one hidden layer")

        layer_sizes = (1,) + tuple(hidden_sizes) + (1,)
        keys = jax.random.split(key, len(layer_sizes) - 1)

        ws = []
        bs = []

        for k, (din, dout) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
            ws.append(weight_scale * jax.random.normal(k, (dout, din)))
            bs.append(jnp.zeros((dout,)))

        self.ws = tuple(ws)
        self.bs = tuple(bs)

        self.k0_raw = jnp.array(K_initial)
        self.which_case = which_case
        self.alpha_corr = jnp.array(alpha_corr)

    def K_mlp(self, eps):
        x = jnp.atleast_1d(eps).reshape(1)

        z = x
        for w, b in zip(self.ws[:-1], self.bs[:-1]):
            z = jax.nn.softplus(w @ z + b)

        k_nn = jax.nn.softplus(self.ws[-1] @ z + self.bs[-1]).squeeze()
        return k_nn

    def K_baseline(self):
        return jax.nn.softplus(self.k0_raw)

    def get_K(self, eps):
        if self.which_case == "only_mlp":
            return self.K_mlp(eps)
        elif self.which_case == "only_baseline":
            return self.K_baseline()
        else:
            return self.alpha_corr*self.K_mlp(eps) + self.K_baseline()

    def __call__(self, eps):
        eps_scalar = jnp.squeeze(eps)
        return self.get_K(eps) * eps_scalar**2


class Signed_MLP_Stiffness(eqx.Module):
    ws: tuple[jax.Array, ...]
    bs: tuple[jax.Array, ...]

    k0_raw: jax.Array
    which_case: str = eqx.field(static=True)

    def __init__(
        self,
        key,
        K_initial=1.0,
        which_case=None,
        hidden_sizes=(10,),
        weight_scale=0.01,
    ):
        if len(hidden_sizes) < 1:
            raise ValueError("Need at least one hidden layer")

        layer_sizes = (1,) + tuple(hidden_sizes) + (1,)
        keys = jax.random.split(key, len(layer_sizes) - 1)

        ws = []
        bs = []

        for k, (din, dout) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
            ws.append(weight_scale * jax.random.normal(k, (dout, din)))
            bs.append(jnp.zeros((dout,)))

        self.ws = tuple(ws)
        self.bs = tuple(bs)

        self.k0_raw = jnp.array(K_initial)
        self.which_case = which_case

    def K_mlp(self, eps):
        x = jnp.atleast_1d(eps).reshape(1)

        z = x
        for w, b in zip(self.ws[:-1], self.bs[:-1]):
            z = jax.nn.softplus(w @ z + b)

        k_nn = (self.ws[-1] @ z + self.bs[-1]).squeeze()
        return k_nn

    def K_baseline(self):
        return jax.nn.softplus(self.k0_raw)

    def get_K(self, eps):
        if self.which_case == "only_mlp":
            return jax.nn.softplus(self.K_mlp(eps))
        elif self.which_case == "only_baseline":
            return self.K_baseline()
        else:
            return jax.nn.softplus(self.K_mlp(eps) + self.K_baseline())

    def __call__(self, eps):
        eps_scalar = jnp.squeeze(eps)
        return self.get_K(eps) * eps_scalar**2
import warnings

import jax
import jax.numpy as jnp
import equinox as eqx

# ===================================================================================== #
# Helpers
# ===================================================================================== #
def inv_softplus(y: jax.Array) -> jax.Array:
    return jnp.log(jnp.expm1(y))


def get_reduced_strain_features(
    del_strain: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    del_strain = jnp.ravel(del_strain)
    if del_strain.shape[0] < 4:
        raise ValueError(
            f"Expected del_strain to have at least 4 entries, got {del_strain.shape}"
        )
    e0 = del_strain[0]
    e1 = del_strain[1]
    eb = del_strain[3]
    return e0, e1, eb


def get_nn_input(
    del_strain: jax.Array,
    input_mode: str,
    only_stretching_NN: bool = False,
    only_bending_NN: bool = False,
) -> jax.Array:
    _validate_nn_feature_flags(only_stretching_NN, only_bending_NN)
    e0, e1, eb = get_reduced_strain_features(del_strain)

    if input_mode == "invariant":
        if only_stretching_NN:
            return jnp.array([e0**2 + e1**2])
        if only_bending_NN:
            return jnp.array([eb**2])
        return jnp.array([e0**2 + e1**2, eb**2])
    if input_mode == "raw":
        if only_stretching_NN:
            return jnp.array([e0, e1])
        if only_bending_NN:
            return jnp.array([eb])
        return jnp.array([e0, e1, eb])

    raise ValueError("input_mode must be 'invariant' or 'raw'")


def _nn_in_features(
    input_mode: str,
    only_stretching_NN: bool = False,
    only_bending_NN: bool = False,
) -> int:
    _validate_nn_feature_flags(only_stretching_NN, only_bending_NN)

    if input_mode == "invariant":
        if only_stretching_NN or only_bending_NN:
            return 1
        return 2
    if input_mode == "raw":
        if only_stretching_NN:
            return 2
        if only_bending_NN:
            return 1
        return 3
    raise ValueError("input_mode must be 'invariant' or 'raw'")


def _validate_nn_feature_flags(
    only_stretching_NN: bool, only_bending_NN: bool
) -> None:
    if only_stretching_NN and only_bending_NN:
        raise ValueError("only_stretching_NN and only_bending_NN cannot both be True.")


def _stiffness_out_features(
    default_out_features: int,
    only_stretching_NN: bool,
    only_bending_NN: bool,
) -> int:
    _validate_nn_feature_flags(only_stretching_NN, only_bending_NN)
    return 1 if only_stretching_NN or only_bending_NN else default_out_features


def _small_linear(in_features: int, out_features: int, key: jax.Array) -> eqx.nn.Linear:
    layer = eqx.nn.Linear(in_features, out_features, key=key)
    return eqx.tree_at(lambda l: l.weight, layer, layer.weight * 1e-2)


def _apply_activation(x: jax.Array, activation: str) -> jax.Array:
    if activation == "softplus":
        return jax.nn.softplus(x)
    if activation == "tanh":
        return jax.nn.tanh(x)
    raise ValueError("activation must be 'softplus' or 'tanh'.")


def _init_diag_raw(der_K: jax.Array) -> jax.Array:
    der_K = jnp.ravel(der_K)
    if der_K.shape != (2,):
        raise ValueError(f"Expected der_K shape (2,), got {der_K.shape}")
    return inv_softplus(jnp.maximum(der_K, 1e-8))


def _init_cholesky_raw(der_K: jax.Array) -> jax.Array:
    der_K = jnp.ravel(der_K)
    if der_K.shape != (3,):
        raise ValueError(f"Expected der_K shape (3,), got {der_K.shape}")

    k_ss0, k_sb0, k_bb0 = der_K
    eps = 1e-6

    if (k_ss0 < 0) or (k_bb0 < 0) or (k_ss0 * k_bb0 - 2.0 * k_sb0**2 < 0):
        raise ValueError(
            "Initial [k_ss, k_sb, k_bb] must satisfy PSD condition: "
            "k_ss >= 0, k_bb >= 0, and k_ss*k_bb - 2*k_sb^2 >= 0."
        )

    l11 = jnp.sqrt(jnp.maximum(k_ss0, eps))
    l21 = jnp.sqrt(2.0) * k_sb0 / l11
    rem = k_bb0 - l21**2
    l22 = jnp.sqrt(jnp.maximum(rem, eps))

    return jnp.array(
        [
            inv_softplus(l11 - eps),
            l21,
            inv_softplus(l22 - eps),
        ]
    )


def _vec_to_L(p: jax.Array) -> jax.Array:
    eps = 1e-6
    p = jnp.ravel(p)
    return jnp.array(
        [
            [jax.nn.softplus(p[0]) + eps, 0.0],
            [p[1], jax.nn.softplus(p[2]) + eps],
        ]
    )


# ===================================================================================== #
# Model params
# ===================================================================================== #
class ModelParams(eqx.Module):
    der_K: jax.Array
    key: jax.Array

    hidden: tuple[int, ...] = eqx.field(static=True, default=(10,))
    which_case: str = eqx.field(static=True, default="MLP")
    corr_factor: float = eqx.field(static=True, default=1.0)
    input_mode: str = eqx.field(static=True, default="raw")
    only_stretching_NN: bool = eqx.field(static=True, default=False)
    only_bending_NN: bool = eqx.field(static=True, default=False)
    zero_reference: bool = eqx.field(static=True, default=True)
    activation: str = eqx.field(static=True, default="softplus")
    mode: str | None = eqx.field(static=True, default=None)


# ===================================================================================== #
# Scalar nets
# ===================================================================================== #
class ScalarMLP(eqx.Module):
    layers: tuple[eqx.nn.Linear, ...]
    positive_output: bool = eqx.field(static=True)
    activation: str = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        hidden: tuple[int, ...],
        key: jax.Array,
        *,
        positive_output: bool,
        activation: str = "softplus",
    ):
        if len(hidden) == 0:
            raise ValueError("hidden must contain at least one hidden layer for ScalarMLP.")

        sizes = (in_features, *hidden, 1)
        keys = jax.random.split(key, len(sizes) - 1)
        self.layers = tuple(
            _small_linear(sizes[i], sizes[i + 1], keys[i])
            for i in range(len(sizes) - 1)
        )
        self.positive_output = positive_output
        self.activation = activation

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.ravel(x)
        for layer in self.layers[:-1]:
            x = _apply_activation(layer(x), self.activation)
        y = self.layers[-1](x)[0]
        return jax.nn.softplus(y) if self.positive_output else y


class ScalarICNN(eqx.Module):
    x_layers: tuple[eqx.nn.Linear, ...]
    z_layers: tuple[eqx.nn.Linear, ...]
    final_x: eqx.nn.Linear
    final_z: eqx.nn.Linear
    positive_output: bool = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        hidden: tuple[int, ...],
        key: jax.Array,
        *,
        positive_output: bool,
    ):
        if len(hidden) == 0:
            raise ValueError("hidden must contain at least one hidden layer.")

        n_x = len(hidden)
        n_z = len(hidden) - 1
        n_total = n_x + n_z + 2
        keys = jax.random.split(key, n_total)

        k = 0
        x_layers = []
        for h in hidden:
            x_layers.append(_small_linear(in_features, h, keys[k]))
            k += 1

        z_layers = []
        for h_in, h_out in zip(hidden[:-1], hidden[1:]):
            z_layers.append(_small_linear(h_in, h_out, keys[k]))
            k += 1

        self.x_layers = tuple(x_layers)
        self.z_layers = tuple(z_layers)
        self.final_x = _small_linear(in_features, 1, keys[k])
        k += 1
        self.final_z = _small_linear(hidden[-1], 1, keys[k])
        self.positive_output = positive_output

    @staticmethod
    def _positive_linear(layer: eqx.nn.Linear, x: jax.Array) -> jax.Array:
        w_pos = jax.nn.softplus(layer.weight)
        return w_pos @ x + layer.bias

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.ravel(x)
        z = jax.nn.softplus(self.x_layers[0](x))
        for x_layer, z_layer in zip(self.x_layers[1:], self.z_layers):
            z = jax.nn.softplus(self._positive_linear(z_layer, z) + x_layer(x))
        y = self._positive_linear(self.final_z, z) + self.final_x(x)
        y = y[0]
        return jax.nn.softplus(y) if self.positive_output else y


class ScalarNet(eqx.Module):
    net: eqx.Module

    def __init__(
        self,
        net_type: str,
        in_features: int,
        hidden: tuple[int, ...],
        key: jax.Array,
        *,
        positive_output: bool,
        activation: str = "softplus",
    ):
        if net_type == "MLP":
            self.net = ScalarMLP(
                in_features=in_features,
                hidden=hidden,
                key=key,
                positive_output=positive_output,
                activation=activation,
            )
        elif net_type == "ICNN":
            self.net = ScalarICNN(
                in_features=in_features,
                hidden=hidden,
                key=key,
                positive_output=positive_output,
            )
        else:
            raise ValueError("net_type must be 'MLP' or 'ICNN'.")

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.net(x)


# ===================================================================================== #
# Shared vector nets
# ===================================================================================== #
class VectorMLP(eqx.Module):
    layers: tuple[eqx.nn.Linear, ...]
    positive_output: bool = eqx.field(static=True)
    activation: str = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        hidden: tuple[int, ...],
        out_features: int,
        key: jax.Array,
        *,
        positive_output: bool,
        activation: str = "softplus",
    ):
        if len(hidden) == 0:
            raise ValueError("hidden must contain at least one hidden layer for VectorMLP.")

        sizes = (in_features, *hidden, out_features)
        keys = jax.random.split(key, len(sizes) - 1)

        self.layers = tuple(
            _small_linear(sizes[i], sizes[i + 1], keys[i])
            for i in range(len(sizes) - 1)
        )
        self.positive_output = positive_output
        self.activation = activation

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.ravel(x)
        for layer in self.layers[:-1]:
            x = _apply_activation(layer(x), self.activation)
        y = self.layers[-1](x)
        return jax.nn.softplus(y) if self.positive_output else y


class VectorICNN(eqx.Module):
    x_layers: tuple[eqx.nn.Linear, ...]
    z_layers: tuple[eqx.nn.Linear, ...]
    final_x: eqx.nn.Linear
    final_z: eqx.nn.Linear
    positive_output: bool = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        hidden: tuple[int, ...],
        out_features: int,
        key: jax.Array,
        *,
        positive_output: bool,
    ):
        if len(hidden) == 0:
            raise ValueError("hidden must contain at least one hidden layer.")

        n_x = len(hidden)
        n_z = len(hidden) - 1
        n_total = n_x + n_z + 2
        keys = jax.random.split(key, n_total)

        k = 0
        x_layers = []
        for h in hidden:
            x_layers.append(_small_linear(in_features, h, keys[k]))
            k += 1

        z_layers = []
        for h_in, h_out in zip(hidden[:-1], hidden[1:]):
            z_layers.append(_small_linear(h_in, h_out, keys[k]))
            k += 1

        self.x_layers = tuple(x_layers)
        self.z_layers = tuple(z_layers)
        self.final_x = _small_linear(in_features, out_features, keys[k]); k += 1
        self.final_z = _small_linear(hidden[-1], out_features, keys[k])
        self.positive_output = positive_output

    @staticmethod
    def _positive_linear(layer: eqx.nn.Linear, x: jax.Array) -> jax.Array:
        w_pos = jax.nn.softplus(layer.weight)
        return w_pos @ x + layer.bias

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.ravel(x)
        z = jax.nn.softplus(self.x_layers[0](x))
        for x_layer, z_layer in zip(self.x_layers[1:], self.z_layers):
            z = jax.nn.softplus(self._positive_linear(z_layer, z) + x_layer(x))
        y = self._positive_linear(self.final_z, z) + self.final_x(x)
        return jax.nn.softplus(y) if self.positive_output else y

class VectorNet(eqx.Module):
    net: eqx.Module

    def __init__(
        self,
        net_type: str,
        in_features: int,
        hidden: tuple[int, ...],
        out_features: int,
        key: jax.Array,
        *,
        positive_output: bool,
        activation: str = "softplus",
    ):
        if net_type == "MLP":
            self.net = VectorMLP(
                in_features=in_features,
                hidden=hidden,
                out_features=out_features,
                key=key,
                positive_output=positive_output,
                activation=activation,
            )
        elif net_type == "ICNN":
            self.net = VectorICNN(
                in_features=in_features,
                hidden=hidden,
                out_features=out_features,
                key=key,
                positive_output=positive_output,
            )
        else:
            raise ValueError("net_type must be 'MLP' or 'ICNN'.")

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.net(x)


# ===================================================================================== #
# 0) ScalarEnergyNN
# ===================================================================================== #
class ScalarEnergyNN(eqx.Module):
    mlp: ScalarNet
    icnn: ScalarNet
    which_case: str = eqx.field(static=True)
    zero_reference: bool = eqx.field(static=True)
    corr_factor: float = eqx.field(static=True)
    input_mode: str = eqx.field(static=True)
    only_stretching_NN: bool = eqx.field(static=True)
    only_bending_NN: bool = eqx.field(static=True)

    def __init__(self, params: ModelParams):
        in_features = _nn_in_features(
            params.input_mode, params.only_stretching_NN, params.only_bending_NN
        )

        self.mlp = ScalarNet(
            "MLP",
            in_features,
            params.hidden,
            params.key,
            positive_output=False,
            activation=params.activation,
        )
        self.icnn = ScalarNet(
            "ICNN",
            in_features,
            params.hidden,
            jax.random.fold_in(params.key, 1),
            positive_output=False,
        )
        self.which_case = params.which_case
        self.zero_reference = params.zero_reference
        self.corr_factor = params.corr_factor
        self.input_mode = params.input_mode
        self.only_stretching_NN = params.only_stretching_NN
        self.only_bending_NN = params.only_bending_NN

    def _net_output(self, x: jax.Array) -> jax.Array:
        if self.which_case == "MLP":
            return self.mlp(x)
        if self.which_case == "ICNN":
            return self.icnn(x)
        raise ValueError("which_case must be 'MLP' or 'ICNN'.")

    def __call__(self, del_strain: jax.Array) -> jax.Array:
        x = get_nn_input(
            del_strain, self.input_mode, self.only_stretching_NN, self.only_bending_NN
        )
        y = self._net_output(x)

        if self.zero_reference:
            y = y - self._net_output(jnp.zeros_like(x))

        # Standalone energy should be exactly zero at the reference and nonnegative.
        return self.corr_factor * y**2
# ===================================================================================== #
# Shared bases
# ===================================================================================== #
class _DiagonalBase:
    K0_raw: jax.Array

    @staticmethod
    def _init_diag(der_K: jax.Array) -> jax.Array:
        return _init_diag_raw(der_K)

    def get_K0(self) -> jax.Array:
        return jax.nn.softplus(self.K0_raw)

    def get_baseline_K_entries(self) -> jax.Array:
        return self.get_K0()

    def _diag_energy_from_entries(
        self, k_s: jax.Array, k_b: jax.Array, del_strain: jax.Array
    ) -> jax.Array:
        e0, e1, eb = get_reduced_strain_features(del_strain)
        return 0.5 * k_s * (e0**2 + e1**2) + 0.5 * k_b * eb**2

    def _diag_matrix(self, k_s: jax.Array, k_b: jax.Array) -> jax.Array:
        return jnp.array(
            [
                [k_s, 0.0, 0.0],
                [0.0, k_s, 0.0],
                [0.0, 0.0, k_b],
            ]
        )


class _CholeskyBase:
    K0_raw: jax.Array

    @staticmethod
    def _init_cholesky(der_K: jax.Array) -> jax.Array:
        return _init_cholesky_raw(der_K)

    def get_B0(self) -> jax.Array:
        L0 = _vec_to_L(self.K0_raw)
        return L0 @ L0.T

    def get_baseline_K_entries(self) -> jax.Array:
        return self._B_to_entries(self.get_B0())

    @staticmethod
    def _B_to_entries(B: jax.Array) -> jax.Array:
        k_ss = B[0, 0]
        k_sb = B[0, 1] / jnp.sqrt(2.0)
        k_bb = B[1, 1]
        return jnp.array([k_ss, k_sb, k_bb])

    @staticmethod
    def _entries_to_matrix(
        k_ss: jax.Array, k_sb: jax.Array, k_bb: jax.Array
    ) -> jax.Array:
        return jnp.array(
            [
                [k_ss, 0.0, k_sb],
                [0.0, k_ss, k_sb],
                [k_sb, k_sb, k_bb],
            ]
        )

    def _chol_energy_from_entries(
        self,
        k_ss: jax.Array,
        k_sb: jax.Array,
        k_bb: jax.Array,
        del_strain: jax.Array,
    ) -> jax.Array:
        e0, e1, eb = get_reduced_strain_features(del_strain)
        return (
            0.5 * k_ss * (e0**2 + e1**2)
            + k_sb * (e0 + e1) * eb
            + 0.5 * k_bb * eb**2
        )


# ===================================================================================== #
# 1) DiagonalPlusEnergyNN
# ===================================================================================== #
class DiagonalPlusEnergyNN(eqx.Module, _DiagonalBase):
    K0_raw: jax.Array
    mlp: ScalarMLP
    icnn: ScalarICNN
    which_case: str = eqx.field(static=True)
    zero_reference: bool = eqx.field(static=True)
    corr_factor: float = eqx.field(static=True)
    input_mode: str = eqx.field(static=True)
    only_stretching_NN: bool = eqx.field(static=True)
    only_bending_NN: bool = eqx.field(static=True)

    def __init__(self, params: ModelParams):
        
        in_features = _nn_in_features(
            params.input_mode, params.only_stretching_NN, params.only_bending_NN
        )

        self.K0_raw = self._init_diag(params.der_K)
        self.mlp = ScalarMLP(in_features, params.hidden, params.key, positive_output=True, activation=params.activation)
        self.icnn = ScalarICNN(in_features, params.hidden, jax.random.fold_in(params.key, 1), positive_output=True)
        self.which_case = params.which_case
        self.zero_reference = params.zero_reference
        self.corr_factor = params.corr_factor
        self.input_mode = params.input_mode
        self.only_stretching_NN = params.only_stretching_NN
        self.only_bending_NN = params.only_bending_NN

    def baseline_energy(self, del_strain: jax.Array) -> jax.Array:
        k_s, k_b = self.get_K0()
        return self._diag_energy_from_entries(k_s, k_b, del_strain)

    def correction_energy(self, del_strain: jax.Array) -> jax.Array:
        x = get_nn_input(
            del_strain, self.input_mode, self.only_stretching_NN, self.only_bending_NN
        )
        net = self.mlp if self.which_case == "MLP" else self.icnn
        out = net(x)
        if self.zero_reference:
            out = out - net(jnp.zeros_like(x))
        return self.corr_factor * out

    def __call__(self, del_strain: jax.Array) -> jax.Array:
        if self.which_case == "baseline":
            return self.baseline_energy(del_strain)
        if self.which_case in ("MLP", "ICNN"):
            return self.baseline_energy(del_strain) + self.correction_energy(del_strain)
        raise ValueError("which_case must be 'baseline', 'MLP', or 'ICNN'.")


# ===================================================================================== #
# 2) CholeskyPlusEnergyNN
# ===================================================================================== #
class CholeskyPlusEnergyNN(eqx.Module, _CholeskyBase):
    K0_raw: jax.Array
    mlp: ScalarMLP
    icnn: ScalarICNN
    which_case: str = eqx.field(static=True)
    zero_reference: bool = eqx.field(static=True)
    corr_factor: float = eqx.field(static=True)
    input_mode: str = eqx.field(static=True)
    only_stretching_NN: bool = eqx.field(static=True)
    only_bending_NN: bool = eqx.field(static=True)

    def __init__(self, params: ModelParams):
        
        in_features = _nn_in_features(
            params.input_mode, params.only_stretching_NN, params.only_bending_NN
        )

        self.K0_raw = self._init_cholesky(params.der_K)
        self.mlp = ScalarMLP(in_features, params.hidden, params.key, positive_output=True, activation=params.activation)
        self.icnn = ScalarICNN(in_features, params.hidden, jax.random.fold_in(params.key, 1), positive_output=True)
        self.which_case = params.which_case
        self.zero_reference = params.zero_reference
        self.corr_factor = params.corr_factor
        self.input_mode = params.input_mode
        self.only_stretching_NN = params.only_stretching_NN
        self.only_bending_NN = params.only_bending_NN

    def get_K_entries(self) -> jax.Array:
        return self._B_to_entries(self.get_B0())

    def get_K_matrix(self) -> jax.Array:
        k_ss, k_sb, k_bb = self.get_K_entries()
        return self._entries_to_matrix(k_ss, k_sb, k_bb)

    def baseline_energy(self, del_strain: jax.Array) -> jax.Array:
        k_ss, k_sb, k_bb = self.get_K_entries()
        return self._chol_energy_from_entries(k_ss, k_sb, k_bb, del_strain)

    def correction_energy(self, del_strain: jax.Array) -> jax.Array:
        x = get_nn_input(
            del_strain, self.input_mode, self.only_stretching_NN, self.only_bending_NN
        )
        net = self.mlp if self.which_case == "MLP" else self.icnn
        out = net(x)
        if self.zero_reference:
            out = out - net(jnp.zeros_like(x))
        return self.corr_factor * out

    def __call__(self, del_strain: jax.Array) -> jax.Array:
        if self.which_case == "baseline":
            return self.baseline_energy(del_strain)
        if self.which_case in ("MLP", "ICNN"):
            return self.baseline_energy(del_strain) + self.correction_energy(del_strain)
        raise ValueError("which_case must be 'baseline', 'MLP', or 'ICNN'.")


# ===================================================================================== #
# 3) DiagonalPlusStiffnessNN  (PSD)
# ===================================================================================== #
class DiagonalPlusStiffnessNN(eqx.Module, _DiagonalBase):
    K0_raw: jax.Array
    mlp: VectorNet
    icnn: VectorNet
    which_case: str = eqx.field(static=True)
    corr_factor: float = eqx.field(static=True)
    input_mode: str = eqx.field(static=True)
    only_stretching_NN: bool = eqx.field(static=True)
    only_bending_NN: bool = eqx.field(static=True)

    def __init__(self, params: ModelParams):
        
        in_features = _nn_in_features(
            params.input_mode, params.only_stretching_NN, params.only_bending_NN
        )
        out_features = _stiffness_out_features(
            2, params.only_stretching_NN, params.only_bending_NN
        )

        self.K0_raw = self._init_diag(params.der_K)
        self.mlp = VectorNet(
            "MLP", in_features, params.hidden, out_features, params.key, positive_output=False, activation=params.activation
        )
        self.icnn = VectorNet(
            "ICNN", in_features, params.hidden, out_features, jax.random.fold_in(params.key, 1), positive_output=False
        )
        self.which_case = params.which_case
        self.corr_factor = params.corr_factor
        self.input_mode = params.input_mode
        self.only_stretching_NN = params.only_stretching_NN
        self.only_bending_NN = params.only_bending_NN

    def get_K_correction(self, del_strain: jax.Array) -> jax.Array:
        x = get_nn_input(
            del_strain, self.input_mode, self.only_stretching_NN, self.only_bending_NN
        )
        raw = self.mlp(x) if self.which_case == "MLP" else self.icnn(x)
        corr = jax.nn.softplus(self.corr_factor * raw)
        if self.only_stretching_NN:
            return jnp.array([corr[0], 0.0])
        if self.only_bending_NN:
            return jnp.array([0.0, corr[0]])
        return corr

    def get_K_total(self, del_strain: jax.Array) -> jax.Array:
        return self.get_K0() + self.get_K_correction(del_strain)

    def get_K_matrix(self, del_strain: jax.Array) -> jax.Array:
        k_s, k_b = self.get_K_total(del_strain)
        return self._diag_matrix(k_s, k_b)

    def __call__(self, del_strain: jax.Array) -> jax.Array:
        k_s, k_b = self.get_K_total(del_strain)
        return self._diag_energy_from_entries(k_s, k_b, del_strain)


# ===================================================================================== #
# 4) CholeskyPlusStiffnessNN  (PSD)
# ===================================================================================== #
class CholeskyPlusStiffnessNN(eqx.Module, _CholeskyBase):
    K0_raw: jax.Array
    mlp: VectorNet
    icnn: VectorNet
    which_case: str = eqx.field(static=True)
    corr_factor: float = eqx.field(static=True)
    input_mode: str = eqx.field(static=True)
    only_stretching_NN: bool = eqx.field(static=True)
    only_bending_NN: bool = eqx.field(static=True)

    def __init__(self, params: ModelParams):
        
        in_features = _nn_in_features(
            params.input_mode, params.only_stretching_NN, params.only_bending_NN
        )
        out_features = _stiffness_out_features(
            3, params.only_stretching_NN, params.only_bending_NN
        )

        self.K0_raw = self._init_cholesky(params.der_K)
        self.mlp = VectorNet(
            "MLP", in_features, params.hidden, out_features, params.key, positive_output=False, activation=params.activation
        )
        self.icnn = VectorNet(
            "ICNN", in_features, params.hidden, out_features, jax.random.fold_in(params.key, 1), positive_output=False
        )
        self.which_case = params.which_case
        self.corr_factor = params.corr_factor
        self.input_mode = params.input_mode
        self.only_stretching_NN = params.only_stretching_NN
        self.only_bending_NN = params.only_bending_NN

    def get_Bnn(self, del_strain: jax.Array) -> jax.Array:
        x = get_nn_input(
            del_strain, self.input_mode, self.only_stretching_NN, self.only_bending_NN
        )
        p = self.mlp(x) if self.which_case == "MLP" else self.icnn(x)
        if self.only_stretching_NN:
            return jnp.array(
                [
                    [jax.nn.softplus(self.corr_factor * p[0]), 0.0],
                    [0.0, 0.0],
                ]
            )
        if self.only_bending_NN:
            return jnp.array(
                [
                    [0.0, 0.0],
                    [0.0, jax.nn.softplus(self.corr_factor * p[0])],
                ]
            )
        L = _vec_to_L(self.corr_factor * p)
        return L @ L.T

    def get_B_total(self, del_strain: jax.Array) -> jax.Array:
        return self.get_B0() + self.get_Bnn(del_strain)

    def get_K_entries(self, del_strain: jax.Array) -> jax.Array:
        return self._B_to_entries(self.get_B_total(del_strain))

    def get_K_matrix(self, del_strain: jax.Array) -> jax.Array:
        k_ss, k_sb, k_bb = self.get_K_entries(del_strain)
        return self._entries_to_matrix(k_ss, k_sb, k_bb)

    def __call__(self, del_strain: jax.Array) -> jax.Array:
        k_ss, k_sb, k_bb = self.get_K_entries(del_strain)
        return self._chol_energy_from_entries(k_ss, k_sb, k_bb, del_strain)


# ===================================================================================== #
# 5) CholeskyPlusStiffnessSignedNN  (signed raw-parameter correction, PSD-guaranteed)
# ===================================================================================== #
class CholeskyPlusStiffnessSignedNN(eqx.Module, _CholeskyBase):
    K0_raw: jax.Array
    mlp: VectorNet
    icnn: VectorNet
    which_case: str = eqx.field(static=True)
    corr_factor: float = eqx.field(static=True)
    input_mode: str = eqx.field(static=True)
    only_stretching_NN: bool = eqx.field(static=True)
    only_bending_NN: bool = eqx.field(static=True)

    def __init__(self, params: ModelParams):
        
        in_features = _nn_in_features(
            params.input_mode, params.only_stretching_NN, params.only_bending_NN
        )
        out_features = _stiffness_out_features(
            3, params.only_stretching_NN, params.only_bending_NN
        )

        self.K0_raw = self._init_cholesky(params.der_K)
        self.mlp = VectorNet(
            "MLP", in_features, params.hidden, out_features, params.key, positive_output=False, activation=params.activation
        )
        self.icnn = VectorNet(
            "ICNN", in_features, params.hidden, out_features, jax.random.fold_in(params.key, 1), positive_output=False
        )
        self.which_case = params.which_case
        self.corr_factor = params.corr_factor
        self.input_mode = params.input_mode
        self.only_stretching_NN = params.only_stretching_NN
        self.only_bending_NN = params.only_bending_NN

    def get_B_total(self, del_strain: jax.Array) -> jax.Array:
        x = get_nn_input(
            del_strain, self.input_mode, self.only_stretching_NN, self.only_bending_NN
        )
        dp = self.mlp(x) if self.which_case == "MLP" else self.icnn(x)
        if self.only_stretching_NN:
            return self.get_B0() + jnp.array(
                [
                    [jax.nn.softplus(self.corr_factor * dp[0]), 0.0],
                    [0.0, 0.0],
                ]
            )
        if self.only_bending_NN:
            return self.get_B0() + jnp.array(
                [
                    [0.0, 0.0],
                    [0.0, jax.nn.softplus(self.corr_factor * dp[0])],
                ]
            )
        L = _vec_to_L(self.K0_raw + self.corr_factor * dp)
        return L @ L.T

    def get_K_entries(self, del_strain: jax.Array) -> jax.Array:
        return self._B_to_entries(self.get_B_total(del_strain))

    def get_K_matrix(self, del_strain: jax.Array) -> jax.Array:
        k_ss, k_sb, k_bb = self.get_K_entries(del_strain)
        return self._entries_to_matrix(k_ss, k_sb, k_bb)

    def __call__(self, del_strain: jax.Array) -> jax.Array:
        k_ss, k_sb, k_bb = self.get_K_entries(del_strain)
        return self._chol_energy_from_entries(k_ss, k_sb, k_bb, del_strain)


#####################################################################################################################
## Brazier's effect:
# ===================================================================================== #
# 6) StructuredBrazierCholeskyEnergyNN
#     Reduced 3-strain energy with explicit Brazier softening in kappa1
#     del_strain = [eps0, eps1, kappa1]
# ===================================================================================== #

def _vec_to_lower_triangular_3(p: jax.Array) -> jax.Array:
    """Map 6-vector to 3x3 lower-triangular matrix."""
    p = jnp.ravel(p)
    if p.shape != (6,):
        raise ValueError(f"Expected p shape (6,), got {p.shape}")

    L = jnp.zeros((3, 3))
    idx = 0
    for i in range(3):
        for j in range(i + 1):
            if i == j:
                L = L.at[i, j].set(jax.nn.softplus(p[idx]) + 1e-6)
            else:
                L = L.at[i, j].set(p[idx])
            idx += 1
    return L


def _reduced_strain_vector(
    del_strain: jax.Array,
) -> jax.Array:
    del_strain = jnp.ravel(del_strain)
    if del_strain.shape == (3,):
        return del_strain

    e0, e1, eb = get_reduced_strain_features(del_strain)
    del_strain = jnp.array([e0, e1, eb])
    if del_strain.shape != (3,):
        raise ValueError(f"Expected reduced del_strain shape (3,), got {del_strain.shape}.")
    return del_strain


def _reduced_strain_nn_input(
    del_strain: jax.Array,
    input_mode: str,
    only_stretching_NN: bool,
    only_bending_NN: bool,
) -> jax.Array:
    del_strain = _reduced_strain_vector(del_strain)
    e0, e1, eb = del_strain

    if input_mode == "invariant":
        if only_stretching_NN:
            return jnp.array([e0**2 + e1**2])
        if only_bending_NN:
            return jnp.array([eb**2])
        return jnp.array([e0**2 + e1**2, eb**2])
    if input_mode == "raw":
        if only_stretching_NN:
            return jnp.array([e0, e1])
        if only_bending_NN:
            return jnp.array([eb])
        return del_strain

    raise ValueError("input_mode must be 'invariant' or 'raw'")


def _reduced_strain_energy_from_K(
    K: jax.Array,
    del_strain: jax.Array,
) -> jax.Array:
    del_strain = _reduced_strain_vector(del_strain)
    return 0.5 * del_strain @ K @ del_strain


def _reduced_strain_nn_in_features(
    input_mode: str,
    only_stretching_NN: bool,
    only_bending_NN: bool,
) -> int:
    return _nn_in_features(input_mode, only_stretching_NN, only_bending_NN)


class StructuredBrazierCholeskyEnergyNN(eqx.Module):
    """
    Energy model for strain vector

        del_strain = [eps0, eps1, kappa1]

    The kappa1 stiffness softens exponentially after a learned critical curvature.
    In anisotropic mode, positive and negative bending directions have different
    thresholds and decay rates. In isotropic mode, they are shared.

    Energy:

        E = 0.5 * eps^T K(eps) eps

    where K(eps) is PSD by construction.
    """

    # Baseline diagonal stiffness for all 3 reduced strain modes
    K0_raw: jax.Array          # shape (3,)

    # Minimum retained stiffness ratio for kappa1 mode
    rho_min_raw: jax.Array     # scalar

    # Critical curvature parameters
    kappa_c_raw: jax.Array     # shape (2,) anisotropic, shape (1,) isotropic

    # Exponential decay rate parameters
    alpha_raw: jax.Array       # shape (2,) anisotropic, shape (1,) isotropic

    # Optional small PSD residual correction
    residual_net: VectorNet

    mode: str = eqx.field(static=True)          # "anisotropic" or "isotropic"
    which_case: str = eqx.field(static=True)    # "baseline", "MLP", or "ICNN"
    corr_factor: float = eqx.field(static=True)
    activation: str = eqx.field(static=True)
    input_mode: str = eqx.field(static=True)
    only_stretching_NN: bool = eqx.field(static=True)
    only_bending_NN: bool = eqx.field(static=True)

    def __init__(
        self,
        params: ModelParams,
        *,
        kappa_c_init: float = 1.0,
        alpha_init: float = 10.0,
        rho_min_init: float = 0.05,
    ):
        mode = params.mode
        if mode is None:
            warnings.warn(
                "StructuredBrazierCholeskyEnergyNN received params.mode=None; "
                "defaulting to isotropic mode.",
                RuntimeWarning,
                stacklevel=2,
            )
            mode = "isotropic"

        if mode not in ("anisotropic", "isotropic"):
            raise ValueError("mode must be either 'anisotropic' or 'isotropic'.")

        self.mode = mode
        self.which_case = params.which_case
        self.corr_factor = params.corr_factor
        self.activation = params.activation
        self.input_mode = params.input_mode
        self.only_stretching_NN = params.only_stretching_NN
        self.only_bending_NN = params.only_bending_NN

        der_K = jnp.ravel(params.der_K)
        if der_K.shape != (3,):
            raise ValueError(
                "StructuredBrazierCholeskyEnergyNN expects params.der_K "
                f"to have shape (3,), got {der_K.shape}."
            )

        self.K0_raw = inv_softplus(jnp.maximum(der_K, 1e-8))

        self.rho_min_raw = inv_softplus(jnp.array(rho_min_init))

        if mode == "anisotropic":
            self.kappa_c_raw = inv_softplus(
                jnp.array([kappa_c_init, kappa_c_init])
            )
            self.alpha_raw = inv_softplus(
                jnp.array([alpha_init, alpha_init])
            )
        else:
            self.kappa_c_raw = inv_softplus(jnp.array([kappa_c_init]))
            self.alpha_raw = inv_softplus(jnp.array([alpha_init]))

        # By default the NN sees the reduced 3D strain vector and outputs a 6-vector
        # parameterizing a 3x3 PSD residual matrix. Single-mode flags make it see
        # only that mode's features and contribute only to that stiffness entry.
        self.residual_net = VectorNet(
            params.which_case if params.which_case in ("MLP", "ICNN") else "MLP",
            in_features=_reduced_strain_nn_in_features(
                params.input_mode, params.only_stretching_NN, params.only_bending_NN
            ),
            hidden=params.hidden,
            out_features=_stiffness_out_features(
                6, params.only_stretching_NN, params.only_bending_NN
            ),
            key=params.key,
            positive_output=False,
            activation=params.activation,
        )

    def get_K0(self) -> jax.Array:
        return jax.nn.softplus(self.K0_raw)

    def get_brazier_params(self):
        rho_min = jax.nn.softplus(self.rho_min_raw)

        # Clamp to avoid rho_min > 1.
        rho_min = jnp.minimum(rho_min, 0.999)

        if self.mode == "anisotropic":
            kappa_c_pos = jax.nn.softplus(self.kappa_c_raw[0])
            kappa_c_neg = jax.nn.softplus(self.kappa_c_raw[1])
            alpha_pos = jax.nn.softplus(self.alpha_raw[0])
            alpha_neg = jax.nn.softplus(self.alpha_raw[1])
        else:
            kappa_c = jax.nn.softplus(self.kappa_c_raw[0])
            alpha = jax.nn.softplus(self.alpha_raw[0])

            kappa_c_pos = kappa_c
            kappa_c_neg = kappa_c
            alpha_pos = alpha
            alpha_neg = alpha

        return rho_min, kappa_c_pos, kappa_c_neg, alpha_pos, alpha_neg

    def brazier_softening_factor(self, kappa1: jax.Array) -> jax.Array:
        """
        Returns s(kappa1) in [rho_min, 1].

        For anisotropic mode:
            positive and negative bending directions can soften at different
            critical curvatures.

        For isotropic mode:
            kappa_c_pos == kappa_c_neg and alpha_pos == alpha_neg.
        """
        rho_min, kappa_c_pos, kappa_c_neg, alpha_pos, alpha_neg = (
            self.get_brazier_params()
        )

        soft_pos = jax.nn.softplus(kappa1 - kappa_c_pos)
        soft_neg = jax.nn.softplus(-kappa1 - kappa_c_neg)

        decay = jnp.exp(
            -alpha_pos * soft_pos**2
            -alpha_neg * soft_neg**2
        )

        return rho_min + (1.0 - rho_min) * decay

    def get_structured_K(self, del_strain: jax.Array) -> jax.Array:
        """
        Structured PSD stiffness matrix.

        Only the kappa1 stiffness is softened explicitly.
        """
        del_strain = _reduced_strain_vector(del_strain)

        K0 = self.get_K0()
        kappa1 = del_strain[2]

        s = self.brazier_softening_factor(kappa1)

        K_diag = K0.at[2].set(K0[2] * s)

        return jnp.diag(K_diag)

    def get_residual_K(self, del_strain: jax.Array) -> jax.Array:
        """
        Small PSD residual correction.

        This lets the NN learn coupling terms and mild corrections while the
        dominant Brazier softening is imposed explicitly.
        """
        if self.which_case == "baseline":
            return jnp.zeros((3, 3))

        x = _reduced_strain_nn_input(
            del_strain, self.input_mode, self.only_stretching_NN, self.only_bending_NN
        )
        p = self.residual_net(x)

        if self.only_stretching_NN:
            K = jnp.zeros((3, 3))
            return K.at[0, 0].set(jax.nn.softplus(self.corr_factor * p[0]))
        if self.only_bending_NN:
            K = jnp.zeros((3, 3))
            return K.at[2, 2].set(jax.nn.softplus(self.corr_factor * p[0]))

        L = _vec_to_lower_triangular_3(self.corr_factor * p)
        return L @ L.T

    def get_K_matrix(self, del_strain: jax.Array) -> jax.Array:
        return self.get_structured_K(del_strain) + self.get_residual_K(del_strain)

    def __call__(self, del_strain: jax.Array) -> jax.Array:
        K = self.get_K_matrix(del_strain)
        return _reduced_strain_energy_from_K(K, del_strain)


# ===================================================================================== #
# 7) StructuredBrazierDiagonalEnergyNN
#     Reduced 3-strain energy with diagonal baseline stiffness and explicit Brazier
#     softening in kappa1
#     del_strain = [eps0, eps1, kappa1]
# ===================================================================================== #
class StructuredBrazierDiagonalEnergyNN(eqx.Module):
    """
    Energy model for strain vector

        del_strain = [eps0, eps1, kappa1]

    Uses a diagonal baseline stiffness:

        E0 = 0.5 * k_s * (eps0^2 + eps1^2) + 0.5 * k_b * kappa1^2

    The bending stiffness k_b softens explicitly after a learned critical
    curvature. An optional diagonal PSD residual can add learned corrections.
    """

    # Baseline diagonal entries: shared stretch stiffness and bending stiffness
    K0_raw: jax.Array          # shape (2,)

    # Minimum retained stiffness ratio for kappa1 mode
    rho_min_raw: jax.Array     # scalar

    # Critical curvature parameters
    kappa_c_raw: jax.Array     # shape (2,) anisotropic, shape (1,) isotropic

    # Exponential decay rate parameters
    alpha_raw: jax.Array       # shape (2,) anisotropic, shape (1,) isotropic

    # Optional small diagonal PSD residual correction
    residual_net: VectorNet

    mode: str = eqx.field(static=True)          # "anisotropic" or "isotropic"
    which_case: str = eqx.field(static=True)    # "baseline", "MLP", or "ICNN"
    corr_factor: float = eqx.field(static=True)
    activation: str = eqx.field(static=True)
    input_mode: str = eqx.field(static=True)
    only_stretching_NN: bool = eqx.field(static=True)
    only_bending_NN: bool = eqx.field(static=True)

    def __init__(
        self,
        params: ModelParams,
        *,
        kappa_c_init: float = 1.0,
        alpha_init: float = 10.0,
        rho_min_init: float = 0.05,
    ):
        mode = params.mode
        if mode is None:
            warnings.warn(
                "StructuredBrazierDiagonalEnergyNN received params.mode=None; "
                "defaulting to isotropic mode.",
                RuntimeWarning,
                stacklevel=2,
            )
            mode = "isotropic"

        if mode not in ("anisotropic", "isotropic"):
            raise ValueError("mode must be either 'anisotropic' or 'isotropic'.")

        self.mode = mode
        self.which_case = params.which_case
        self.corr_factor = params.corr_factor
        self.activation = params.activation
        self.input_mode = params.input_mode
        self.only_stretching_NN = params.only_stretching_NN
        self.only_bending_NN = params.only_bending_NN

        der_K = jnp.ravel(params.der_K)
        if der_K.shape != (2,):
            raise ValueError(
                "StructuredBrazierDiagonalEnergyNN expects params.der_K "
                f"to have shape (2,), got {der_K.shape}."
            )

        self.K0_raw = inv_softplus(jnp.maximum(der_K, 1e-8))
        self.rho_min_raw = inv_softplus(jnp.array(rho_min_init))

        if mode == "anisotropic":
            self.kappa_c_raw = inv_softplus(
                jnp.array([kappa_c_init, kappa_c_init])
            )
            self.alpha_raw = inv_softplus(
                jnp.array([alpha_init, alpha_init])
            )
        else:
            self.kappa_c_raw = inv_softplus(jnp.array([kappa_c_init]))
            self.alpha_raw = inv_softplus(jnp.array([alpha_init]))

        self.residual_net = VectorNet(
            params.which_case if params.which_case in ("MLP", "ICNN") else "MLP",
            in_features=_reduced_strain_nn_in_features(
                params.input_mode, params.only_stretching_NN, params.only_bending_NN
            ),
            hidden=params.hidden,
            out_features=_stiffness_out_features(
                2, params.only_stretching_NN, params.only_bending_NN
            ),
            key=params.key,
            positive_output=False,
            activation=params.activation,
        )

    def get_K0(self) -> jax.Array:
        return jax.nn.softplus(self.K0_raw)

    def get_brazier_params(self):
        rho_min = jax.nn.softplus(self.rho_min_raw)
        rho_min = jnp.minimum(rho_min, 0.999)

        if self.mode == "anisotropic":
            kappa_c_pos = jax.nn.softplus(self.kappa_c_raw[0])
            kappa_c_neg = jax.nn.softplus(self.kappa_c_raw[1])
            alpha_pos = jax.nn.softplus(self.alpha_raw[0])
            alpha_neg = jax.nn.softplus(self.alpha_raw[1])
        else:
            kappa_c = jax.nn.softplus(self.kappa_c_raw[0])
            alpha = jax.nn.softplus(self.alpha_raw[0])

            kappa_c_pos = kappa_c
            kappa_c_neg = kappa_c
            alpha_pos = alpha
            alpha_neg = alpha

        return rho_min, kappa_c_pos, kappa_c_neg, alpha_pos, alpha_neg

    def brazier_softening_factor(self, kappa1: jax.Array) -> jax.Array:
        rho_min, kappa_c_pos, kappa_c_neg, alpha_pos, alpha_neg = (
            self.get_brazier_params()
        )

        soft_pos = jax.nn.softplus(kappa1 - kappa_c_pos)
        soft_neg = jax.nn.softplus(-kappa1 - kappa_c_neg)

        decay = jnp.exp(
            -alpha_pos * soft_pos**2
            -alpha_neg * soft_neg**2
        )

        return rho_min + (1.0 - rho_min) * decay

    def get_structured_K(self, del_strain: jax.Array) -> jax.Array:
        """
        Diagonal PSD stiffness matrix with explicit bending softening.
        """
        del_strain = _reduced_strain_vector(del_strain)

        k_s, k_b = self.get_K0()
        s = self.brazier_softening_factor(del_strain[2])

        return jnp.diag(jnp.array([k_s, k_s, k_b * s]))

    def get_residual_K(self, del_strain: jax.Array) -> jax.Array:
        """
        Small diagonal PSD residual correction.
        """
        if self.which_case == "baseline":
            return jnp.zeros((3, 3))

        x = _reduced_strain_nn_input(
            del_strain, self.input_mode, self.only_stretching_NN, self.only_bending_NN
        )
        p = self.residual_net(x)
        corr = jax.nn.softplus(self.corr_factor * p)

        if self.only_stretching_NN:
            return jnp.diag(jnp.array([corr[0], corr[0], 0.0]))
        if self.only_bending_NN:
            return jnp.diag(jnp.array([0.0, 0.0, corr[0]]))
        return jnp.diag(jnp.array([corr[0], corr[0], corr[1]]))

    def get_K_matrix(self, del_strain: jax.Array) -> jax.Array:
        return self.get_structured_K(del_strain) + self.get_residual_K(del_strain)

    def __call__(self, del_strain: jax.Array) -> jax.Array:
        K = self.get_K_matrix(del_strain)
        return _reduced_strain_energy_from_K(K, del_strain)


# ===================================================================================== #
# Cholesky baseline stiffness with explicit exponential stretch correction
# ===================================================================================== #
class CholeskyPlusExponentialStretchStiffness(eqx.Module, _CholeskyBase):
    K0_raw: jax.Array
    corr_factor: float = eqx.field(static=True)
    alpha: jax.Array # learnable

    def __init__(self, params: ModelParams):
        self.K0_raw = self._init_cholesky(params.der_K)
        self.corr_factor = params.corr_factor
        self.alpha = jax.numpy.array(1.0)  # Start with a small correction; can be learned to grow with stretch.

    def get_K_correction(self, del_strain: jax.Array) -> jax.Array:
        e0, e1, _ = get_reduced_strain_features(del_strain)
        # normalize
        mean_stretch = (e0 + e1) / 2.0
        mean_stretch_normalized = mean_stretch / 10.0  # 10 is a rough scale for large stretch
        # Exponential correction that grows with the normalized stretch, minus 1 to make it zero at the reference.
        k_ss_corr = self.corr_factor * (jnp.exp(self.alpha * mean_stretch_normalized)-1.0)
        return jnp.array([k_ss_corr, 0.0, 0.0])

    def get_K_entries(self, del_strain: jax.Array) -> jax.Array:
        return self.get_baseline_K_entries() + self.get_K_correction(del_strain)

    def get_K_matrix(self, del_strain: jax.Array) -> jax.Array:
        k_ss, k_sb, k_bb = self.get_K_entries(del_strain)
        return self._entries_to_matrix(k_ss, k_sb, k_bb)

    def __call__(self, del_strain: jax.Array) -> jax.Array:
        k_ss, k_sb, k_bb = self.get_K_entries(del_strain)
        return self._chol_energy_from_entries(k_ss, k_sb, k_bb, del_strain)

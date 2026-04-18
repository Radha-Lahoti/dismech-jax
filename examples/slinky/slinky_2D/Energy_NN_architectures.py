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


def get_nn_input(del_strain: jax.Array, input_mode: str) -> jax.Array:
    e0, e1, eb = get_reduced_strain_features(del_strain)

    if input_mode == "invariant":
        return jnp.array([e0**2 + e1**2, eb**2])
    if input_mode == "raw":
        return jnp.array([e0, e1, eb])

    raise ValueError("input_mode must be 'invariant' or 'raw'")


def _nn_in_features(input_mode: str) -> int:
    if input_mode == "invariant":
        return 2
    if input_mode == "raw":
        return 3
    raise ValueError("input_mode must be 'invariant' or 'raw'")


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
    zero_reference: bool = eqx.field(static=True, default=True)
    activation: str = eqx.field(static=True, default="softplus")


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

    def __init__(self, params: ModelParams):
        in_features = _nn_in_features(params.input_mode)

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

    def _net_output(self, x: jax.Array) -> jax.Array:
        if self.which_case == "MLP":
            return self.mlp(x)
        if self.which_case == "ICNN":
            return self.icnn(x)
        raise ValueError("which_case must be 'MLP' or 'ICNN'.")

    def __call__(self, del_strain: jax.Array) -> jax.Array:
        x = get_nn_input(del_strain, self.input_mode)
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

    def __init__(self, params: ModelParams):
        
        in_features = _nn_in_features(params.input_mode)

        self.K0_raw = self._init_diag(params.der_K)
        self.mlp = ScalarMLP(in_features, params.hidden, params.key, positive_output=True, activation=params.activation)
        self.icnn = ScalarICNN(in_features, params.hidden, jax.random.fold_in(params.key, 1), positive_output=True)
        self.which_case = params.which_case
        self.zero_reference = params.zero_reference
        self.corr_factor = params.corr_factor
        self.input_mode = params.input_mode

    def baseline_energy(self, del_strain: jax.Array) -> jax.Array:
        k_s, k_b = self.get_K0()
        return self._diag_energy_from_entries(k_s, k_b, del_strain)

    def correction_energy(self, del_strain: jax.Array) -> jax.Array:
        x = get_nn_input(del_strain, self.input_mode)
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

    def __init__(self, params: ModelParams):
        
        in_features = _nn_in_features(params.input_mode)

        self.K0_raw = self._init_cholesky(params.der_K)
        self.mlp = ScalarMLP(in_features, params.hidden, params.key, positive_output=True, activation=params.activation)
        self.icnn = ScalarICNN(in_features, params.hidden, jax.random.fold_in(params.key, 1), positive_output=True)
        self.which_case = params.which_case
        self.zero_reference = params.zero_reference
        self.corr_factor = params.corr_factor
        self.input_mode = params.input_mode

    def get_K_entries(self) -> jax.Array:
        return self._B_to_entries(self.get_B0())

    def get_K_matrix(self) -> jax.Array:
        k_ss, k_sb, k_bb = self.get_K_entries()
        return self._entries_to_matrix(k_ss, k_sb, k_bb)

    def baseline_energy(self, del_strain: jax.Array) -> jax.Array:
        k_ss, k_sb, k_bb = self.get_K_entries()
        return self._chol_energy_from_entries(k_ss, k_sb, k_bb, del_strain)

    def correction_energy(self, del_strain: jax.Array) -> jax.Array:
        x = get_nn_input(del_strain, self.input_mode)
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

    def __init__(self, params: ModelParams):
        
        in_features = _nn_in_features(params.input_mode)

        self.K0_raw = self._init_diag(params.der_K)
        self.mlp = VectorNet(
            "MLP", in_features, params.hidden, 2, params.key, positive_output=False, activation=params.activation
        )
        self.icnn = VectorNet(
            "ICNN", in_features, params.hidden, 2, jax.random.fold_in(params.key, 1), positive_output=False
        )
        self.which_case = params.which_case
        self.corr_factor = params.corr_factor
        self.input_mode = params.input_mode

    def get_K_correction(self, del_strain: jax.Array) -> jax.Array:
        x = get_nn_input(del_strain, self.input_mode)
        raw = self.mlp(x) if self.which_case == "MLP" else self.icnn(x)
        return jax.nn.softplus(self.corr_factor * raw)

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

    def __init__(self, params: ModelParams):
        
        in_features = _nn_in_features(params.input_mode)

        self.K0_raw = self._init_cholesky(params.der_K)
        self.mlp = VectorNet(
            "MLP", in_features, params.hidden, 3, params.key, positive_output=False, activation=params.activation
        )
        self.icnn = VectorNet(
            "ICNN", in_features, params.hidden, 3, jax.random.fold_in(params.key, 1), positive_output=False
        )
        self.which_case = params.which_case
        self.corr_factor = params.corr_factor
        self.input_mode = params.input_mode

    def get_Bnn(self, del_strain: jax.Array) -> jax.Array:
        x = get_nn_input(del_strain, self.input_mode)
        p = self.mlp(x) if self.which_case == "MLP" else self.icnn(x)
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

    def __init__(self, params: ModelParams):
        
        in_features = _nn_in_features(params.input_mode)

        self.K0_raw = self._init_cholesky(params.der_K)
        self.mlp = VectorNet(
            "MLP", in_features, params.hidden, 3, params.key, positive_output=False, activation=params.activation
        )
        self.icnn = VectorNet(
            "ICNN", in_features, params.hidden, 3, jax.random.fold_in(params.key, 1), positive_output=False
        )
        self.which_case = params.which_case
        self.corr_factor = params.corr_factor
        self.input_mode = params.input_mode

    def get_B_total(self, del_strain: jax.Array) -> jax.Array:
        x = get_nn_input(del_strain, self.input_mode)
        dp = self.mlp(x) if self.which_case == "MLP" else self.icnn(x)
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

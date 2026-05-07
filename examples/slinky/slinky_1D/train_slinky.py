"""
Train an ICNN energy model on 1D slinky pull data (same script as before, shared core in ``displacement_experiment``).
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import matplotlib.pyplot as plt

from Energy_NN import ICNN_Energy
from displacement_experiment import DisplacementForceTrainer

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Load experiment NPZ and build trainer (data split + optimizer live in the class)
# ---------------------------------------------------------------------------
trainer, slinky, data = DisplacementForceTrainer.from_npz("slinky_pulling_force_data.npz")
split = trainer.split

model = ICNN_Energy(jax.random.PRNGKey(42), K_initial=1)

opt_state = trainer.optimizer.init(eqx.filter(model, eqx.is_inexact_array))

num_epochs = 10000
log_freq = 500

print("Num total samples:", data.lambdas.shape[0])
print("Num train samples:", split.train_lambdas.shape[0])
print("Num test samples :", split.test_lambdas.shape[0])
print("Initial pulled-node x:", data.initial_last_node_x)
print("Final pulled-node x  :", data.final_last_node_x)
print("Rest length l_k0     :", float(data.l_k0))

final_model, loss_history = trainer.train_loop(
    model, opt_state, num_epochs=num_epochs, log_freq=log_freq
)

# ---------------------------------------------------------------------------
# Full-trajectory metrics
# ---------------------------------------------------------------------------
lambdas = data.lambdas
force_truth = data.force_truth
pred_full_force = trainer.predict_force(final_model, lambdas)

train_mse = jnp.mean((force_truth[split.train_mask] - pred_full_force[split.train_mask]) ** 2)
test_mse = jnp.mean((force_truth[split.test_mask] - pred_full_force[split.test_mask]) ** 2)

print("Final train force MSE:", train_mse)
print("Final test  force MSE:", test_mse)

# ---------------------------------------------------------------------------
# Plots (unchanged presentation; uses shared ``SlinkyPullNpzData`` fields)
# ---------------------------------------------------------------------------
pulled_node_x = (1.0 - lambdas) * data.initial_last_node_x + lambdas * data.final_last_node_x

lambdas_np = np.array(lambdas)
force_truth_np = np.array(force_truth)
pred_full_force_np = np.array(pred_full_force)
pulled_node_x_np = np.array(pulled_node_x)

train_mask_np = np.array(split.train_mask)
test_mask_np = np.array(split.test_mask)

plt.figure(figsize=(7, 5))
plt.plot(pulled_node_x_np, pred_full_force_np, "--", linewidth=2, label="Prediction")
plt.plot(
    pulled_node_x_np[train_mask_np],
    force_truth_np[train_mask_np],
    "o",
    markersize=6,
    label="Train samples",
)
plt.plot(
    pulled_node_x_np[test_mask_np],
    force_truth_np[test_mask_np],
    "s",
    markersize=6,
    label="Test samples",
)
plt.xlabel("Pulled node x")
plt.ylabel("Force")
plt.title("Force vs pulled-node displacement")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

qs = jax.vmap(lambda lam: slinky.get_q(lam, data.q0))(lambdas)
strains = jax.vmap(slinky.get_eps)(qs)
energy = jax.vmap(lambda eps: final_model(jnp.array([eps])))(strains)

plt.figure(figsize=(7, 5))
plt.plot(np.array(strains), np.array(energy), linewidth=2, label="Learned energy")
plt.xlabel("Strain")
plt.ylabel("Elastic energy")
plt.title("Elastic energy vs strain")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

stiffness = jax.vmap(
    lambda eps: jax.grad(lambda e: jax.grad(final_model)(jnp.array([e]))[0])(eps)
)(strains)

plt.figure(figsize=(7, 5))
plt.plot(np.array(strains), np.array(stiffness), linewidth=2, label="Learned stiffness")
plt.xlabel("Strain")
plt.ylabel("Effective stiffness")
plt.title("Effective stiffness vs strain (ICNN)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

loss_history_np = np.array(loss_history)
train_hist = loss_history_np[0, :]
test_hist = loss_history_np[1, :]

plt.figure()
plt.plot(train_hist, label="Train loss")
plt.plot(test_hist, label="Test loss")
plt.yscale("log")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Test loss")
plt.show()

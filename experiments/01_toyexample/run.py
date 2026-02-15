import sys
from pathlib import Path

import hydra
import jax
import omegaconf
import pandas as pd
from jax import numpy as jnp

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import plot_loss_grads

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def step_ideal(q, v, h):  # Ideal elastic collision simulation
    a = 0.0  # Contact-free acceleration
    q, v = jax.lax.cond(q < 0.0, lambda: (q, -v), lambda: (q, v))
    v = v + a * h  # Semi-implicit Euler
    q = q + v * h
    return q, v


def step_penalty(q, v, h):  # Penalty based collision simulation
    a = 0.0  # Contact-free acceleration
    a_ref = jax.lax.cond(q < 0.0, lambda: -1.0 * v - 50.0 * q, lambda: 0.0)
    v = v + (a + a_ref) * h  # Semi-implicit Euler
    q = q + v * h
    return q, v


@jax.value_and_grad
def unroll(q0, v0, n_steps, step_fn, h):
    def body(carry, _):
        q, v = carry
        q, v = step_fn(q, v, h)
        return (q, v), (q, v)

    (qf, vf), _ = jax.lax.scan(body, (q0, v0), length=n_steps)
    q_target = 1.0
    loss = jnp.abs(qf - q_target)
    return loss


@hydra.main(config_path=".", config_name="config", version_base="1.3")
def main(cfg):
    print(f"Starting run with parameters:\n{omegaconf.OmegaConf.to_yaml(cfg)}")

    v0 = jnp.array(cfg.v0)
    q0_values = jnp.linspace(cfg.q0_min, cfg.q0_max, cfg.q0_resolution)

    # Coarse simulation
    h = cfg.coarse.h
    nsteps = cfg.coarse.nsteps
    step_ideal_vmap = jax.vmap(lambda q0: unroll(q0, v0, nsteps, step_ideal, h))
    step_penalty_vmap = jax.vmap(lambda q0: unroll(q0, v0, nsteps, step_penalty, h))
    loss_ideal, grad_ideal = step_ideal_vmap(q0_values)
    loss_penalty, grad_penalty = step_penalty_vmap(q0_values)

    # Fine simulation
    h = cfg.fine.h
    nsteps = cfg.fine.nsteps
    step_ideal_vmap2 = jax.vmap(lambda q0: unroll(q0, v0, nsteps, step_ideal, h))
    step_penalty_vmap2 = jax.vmap(lambda q0: unroll(q0, v0, nsteps, step_penalty, h))
    loss_ideal2, grad_ideal2 = step_ideal_vmap2(q0_values)
    loss_penalty2, grad_penalty2 = step_penalty_vmap2(q0_values)

    # Save raw data
    df = pd.DataFrame({
        "q0": q0_values,
        "loss_ideal": loss_ideal,
        "grad_ideal": grad_ideal,
        "loss_penalty": loss_penalty,
        "grad_penalty": grad_penalty,
        "loss_ideal_fine": loss_ideal2,
        "grad_ideal_fine": grad_ideal2,
        "loss_penalty_fine": loss_penalty2,
        "grad_penalty_fine": grad_penalty2,
    })
    df.to_csv(RESULTS_DIR / "data.csv", index=False)
    print(f"Data saved to {RESULTS_DIR / 'data.csv'}")

    plot_loss_grads(
        q0_values, loss_ideal, grad_ideal,
        str(RESULTS_DIR / "ideal_elastic"),
        xlabel=r"Initial position $q_0$",
    )
    plot_loss_grads(
        q0_values, loss_penalty, grad_penalty,
        str(RESULTS_DIR / "penalty_based"),
        xlabel=r"Initial position $q_0$",
    )
    plot_loss_grads(
        q0_values, loss_ideal2, grad_ideal2,
        str(RESULTS_DIR / "ideal_elastic_fine"),
        xlabel=r"Initial position $q_0$",
    )
    plot_loss_grads(
        q0_values, loss_penalty2, grad_penalty2,
        str(RESULTS_DIR / "penalty_based_fine"),
        xlabel=r"Initial position $q_0$",
    )


if __name__ == "__main__":
    main()

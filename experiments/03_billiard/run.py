# Code adapted from: https://github.com/google-deepmind/mujoco/issues/2237
# Simple optimization problem where a ball is thrown at another ball.
# The first ball shall be thrown in such that the second ball stops at a certain position.

import sys
from os.path import join
from pathlib import Path
from tqdm import tqdm
import hydra
import omegaconf
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import equinox as eqx
import optax
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "high")
jax.config.update("jax_disable_jit", False)


# Add parent directory for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    cfg_to_flat_dct, make_run_dir, plot_loss_grads,
    render_trajectory, save_run_metadata, upscale,
)

import mujoco
from mujoco import mjx
import mjx_diffrax

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def make_qfrcs(u0, Nlength, nv):
    """Build the qfrcs_applied array and apply force at first timestep."""
    qfrcs_applied = jnp.zeros((Nlength, nv))
    qfrcs_applied = qfrcs_applied.at[0, 0].set(4.0)
    qfrcs_applied = qfrcs_applied.at[0, 1].set(u0)
    return qfrcs_applied


@eqx.filter_jit
@eqx.filter_value_and_grad
def loss(u0, m, d, Nlength, r_cost_weight, cfg_diffrax):
    """Sum of running costs and final cost."""
    qfrcs = make_qfrcs(u0, Nlength, m.nv)
    d, ds = mjx_diffrax.multistep(m, d, nsteps=Nlength, cfg=cfg_diffrax, qfrcs_applied=qfrcs)

    def cost(d):
        # Distance of second ball to origin
        return jnp.sum(d.qpos[7:10]**2) 

    return r_cost_weight * jnp.sum(jax.vmap(cost)(ds), axis=0) + cost(d)


def render_u0(u0, m, d, mj_model, Nlength, evalname, camera, cfg_diffrax, run_dir):
    """Visualisation for a single force."""
    qfrcs = make_qfrcs(u0, Nlength, m.nv)
    _, traj_data = mjx_diffrax.multistep(m, d, nsteps=Nlength, cfg=cfg_diffrax, qfrcs_applied=qfrcs)
    render_trajectory(
        traj=traj_data,
        m=mj_model,
        path=str(run_dir),
        name=evalname,
        camera=camera,
        reduce_video_size=True,
    )


def train(
    m,
    d,
    mj_model,
    u0,
    max_iter,
    Nlength,
    start_learning_rate,
    tol,
    r_cost_weight,
    qx0,
    qx1,
    camera,
    cfg_diffrax,
    run_dir,
    render=True,
):
    # Initial conditions
    d = d.replace(qpos=d.qpos.at[0].set(qx0).at[7].set(qx1))

    # Render initial control
    if render:
        render_u0(
            u0,
            m,
            d,
            mj_model,
            Nlength,
            evalname=f"billiard_before_training_u0={u0}",
            camera=camera,
            cfg_diffrax=cfg_diffrax,
            run_dir=run_dir,
        )

    # Gradient descent
    optimizer = optax.adam(start_learning_rate)
    opt_state = optimizer.init(u0)
    filter_spec = u0

    u0_best = u0
    loss_best = float("inf")
    for _ in tqdm(range(max_iter)):
        u0_prev = u0
        lossval, grads = loss(u0, m, d, Nlength, r_cost_weight, cfg_diffrax)
        updates, opt_state = optimizer.update(grads, opt_state, filter_spec)
        u0 = optax.apply_updates(u0, updates)
        tqdm.write(f"Loss: {lossval:.3f}, u: {u0:.3f}")

        if lossval < loss_best:
            loss_best = lossval
            u0_best = u0

        if abs(u0 - u0_prev) < tol or jnp.isnan(grads).any():
            break

    # Render best control
    if render:
        render_u0(
            u0_best,
            m,
            d,
            mj_model,
            Nlength,
            evalname=f"billiard_after_training_u0={u0_best}",
            camera=camera,
            cfg_diffrax=cfg_diffrax,
            run_dir=run_dir,
        )


def analyze(
    m,
    d,
    mj_model,
    u0,
    u0_min,
    u0_max,
    u0_resolution,
    Nlength,
    r_cost_weight,
    qx0,
    qx1,
    camera,
    cfg_diffrax,
    run_dir,
    render=True,
):
    d = d.replace(qpos=d.qpos.at[0].set(qx0).at[7].set(qx1))
    u0s = jnp.linspace(u0_min, u0_max, u0_resolution)

    if render:
        render_u0(u0, m, d, mj_model, Nlength, evalname=f"billiard_analyze_u0={u0}", camera=camera, cfg_diffrax=cfg_diffrax, run_dir=run_dir)
        render_u0(u0s[0], m, d, mj_model, Nlength, evalname=f"billiard_analyze_u0={u0s[0]}", camera=camera, cfg_diffrax=cfg_diffrax, run_dir=run_dir)
        render_u0(u0s[-1], m, d, mj_model, Nlength, evalname=f"billiard_analyze_u0={u0s[-1]}", camera=camera, cfg_diffrax=cfg_diffrax, run_dir=run_dir)

    loss_vmap = eqx.filter_vmap(loss, in_axes=(0, None, None, None, None, None))
    losses, grads = loss_vmap(u0s, m, d, Nlength, r_cost_weight, cfg_diffrax)

    u0s, losses, grads = np.array(u0s), np.array(losses), np.array(grads)
    pd.DataFrame({
        "u0s": u0s,
        "losses": losses,
        "grads": grads,
    }).to_csv(run_dir / "billiard_analyze.csv", index=False)
    print(f"Data saved to {run_dir / 'billiard_analyze.csv'}")

    plot_loss_grads(u0s, losses, grads, figpath=str(run_dir / "billiard_analyze"))


@hydra.main(config_path=".", config_name="config", version_base="1.3")
def main(cfg):
    if cfg.quick:
        omegaconf.OmegaConf.update(cfg, "general.Nlength", 10)
        omegaconf.OmegaConf.update(cfg, "analyze.u0_resolution", 5)
        omegaconf.OmegaConf.update(cfg, "train.max_iter", 2)
    print(f"Starting run with parameters: \n{omegaconf.OmegaConf.to_yaml(cfg)}")

    # Create timestamped results directory
    descriptor = cfg.get("run_name", f"{cfg.xml.system}_{cfg.mode}")
    run_dir = make_run_dir(RESULTS_DIR, descriptor)
    save_run_metadata(run_dir, cfg)
    print(f"Results directory: {run_dir}")

    local_path = join(cfg.xml.path, cfg.xml.system + ".xml")

    # Load mj and mjx model
    mj_model = mujoco.MjModel.from_xml_path(local_path)

    m = mjx.put_model(mj_model)
    if m.mesh_convex == ():
        m = m.replace(mesh_convex=None)  # empty tuple breaks adjoint gradients in diffrax
    m = m.tree_replace(cfg_to_flat_dct(cfg.xml.overwrite))
    mj_model.opt.timestep = float(m.opt.timestep)
    
    d = mjx.make_data(m)
    d = jax.tree.map(upscale, d)

    diffrax_cfg = mjx_diffrax.DiffraxConfig(**cfg.diffrax)

    if cfg.mode == "train":
        train(m, d, mj_model, **cfg.train, **cfg.general, cfg_diffrax=diffrax_cfg, run_dir=run_dir, render=cfg.render)
    elif cfg.mode == "analyze":
        analyze(m, d, mj_model, **cfg.analyze, **cfg.general, cfg_diffrax=diffrax_cfg, run_dir=run_dir, render=cfg.render)


if __name__ == "__main__":
    main()
